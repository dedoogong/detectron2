# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from detectron2.modeling.detector.yolact import PredictionModule
from detectron2.modeling.detector.layers import Detect
from detectron2.modeling.detector.utils import timer
from detectron2.modeling.detector.layers.interpolate import InterpolateModule
import numpy as np
import torch.nn.functional as F
from projects.Yolact.config import cfg as config
from detectron2.layers.modules.multibox_loss import MultiBoxLoss
criterion = MultiBoxLoss(num_classes=config.num_classes,
                             pos_threshold=config.positive_iou_threshold,
                             neg_threshold=config.negative_iou_threshold,
                             negpos_ratio=config.ohem_negpos_ratio)

__all__ = ["GeneralizedRCNN", "ProposalNetwork", "YolactBackboneWithFPN"]


class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params

    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)

def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """
    def make_layer(layer_cfg):
        nonlocal in_channels

        # Possible patterns:
        # ( 256, 3, {}) -> conv
        # ( 256,-2, {}) -> deconv
        # (None,-2, {}) -> bilinear interpolate
        # ('cat',[],{}) -> concat the subnetworks in the list
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])

        in_channels = num_channels if num_channels is not None else in_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat together all the component layer lists
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels

@META_ARCH_REGISTRY.register()
class YolactBackboneWithFPN(nn.Module):

    def freeze_bn(self):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if self.cfg.fpn is not None and int(key.split('.')[2]) >= self.cfg.fpn.num_downsample:
                    del state_dict[key]

        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')

        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = isinstance(module, torch.jit.WeakScriptModuleProxy) \
                             and all_in(module.__dict__['_constants_set'], conv_constants) \
                             and all_in(conv_constants, module.__dict__['_constants_set'])

            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    if self.cfg.use_focal_loss and 'conf_layer' in name:
                        if not self.cfg.use_sigmoid_focal_loss:
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0
                            # self.cfg.focal_loss_init_pi==0.01
                            module.bias.data[0]  = np.log((1 - self.cfg.focal_loss_init_pi) / self.cfg.focal_loss_init_pi)
                            #self.backbone.size_divisibility
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0]  = -np.log(self.cfg.focal_loss_init_pi / (1 - self.cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - self.cfg.focal_loss_init_pi) / self.cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()

    def __init__(self, cfg_maskrcnn):
        super(YolactBackboneWithFPN, self).__init__()
        cfg = cfg_maskrcnn
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg) # build backbone using default detectron2's settings

        #self.freeze_bn() maybe it is alreay done in above step?? check!


        ## Yolact start ##
        self.cfg = config
        #self.rpn = build_rpn(cfg, self.backbone.out_channels)
        #self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        ################################################################

        if self.cfg.mask_type == 1 : #mask_type.lincomb:
            if self.cfg.mask_proto_use_grid:
                self.grid = torch.Tensor(np.load(self.cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0

            self.proto_src = self.cfg.mask_proto_src

            if self.proto_src is None: in_channels = 3
            elif self.cfg.fpn is not None: in_channels = self.cfg.fpn.num_features
            else: in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids

            # The include_last_relu=false here is because we might want to change it to another function
            self.proto_net, self.cfg.mask_dim = make_net(in_channels, self.cfg.mask_proto_net, include_last_relu=False)

            if self.cfg.mask_proto_bias:
                self.cfg.mask_dim += 1

        else :# self.cfg.mask_type == self.cfg.mask_type.direct:
            self.cfg.mask_dim = self.cfg.mask_size**2


        #self.backbone_yolact = construct_backbone(self.cfg.backbone)
        self.selected_layers = [1,2,3]#self.cfg.backbone.selected_layers
        src_channels = [256,512,1024,2048] # self.backbone.channels

        #####################################

        self.prediction_layers = nn.ModuleList()
        self.selected_layers = [0,1,2,3,4]   # list(range(len(self.selected_layers) + self.cfg.fpn.num_downsample))#
        src_channels = [256,256,256,256,256] # [self.cfg.fpn.num_features] * len(self.selected_layers)#
        pred_aspect_ratios=[[[1, 0.5, 2]],[[1, 0.5, 2]],[[1, 0.5, 2]],[[1, 0.5, 2]],[[1, 0.5, 2]]]
        pred_scales=[[24],[48],[96],[192],[384]]
        self.num_classes = 81
        self.mask_type_lincomb = 1
        self.mask_type = self.mask_type_lincomb
        self.eval_mask_branch = True
        self.mask_proto_prototypes_as_features=False
        self.share_prediction_module= True
        self.use_instance_coeff=False
        self.mask_proto_prototype_activation=lambda x: torch.nn.functional.relu(x, inplace=True)#torch.nn.ReLU
        self.mask_proto_prototypes_as_features=False
        self.mask_proto_bias=False
        #self.mask_proto_prototype_activation
        self.use_class_existence_loss = False
        self.use_semantic_segmentation_loss = True
        self.mask_proto_prototypes_as_features_no_grad=False

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if True and idx > 0:# self.cfg.share_prediction_module and=True
                parent = self.prediction_layers[0]
            #idx, layer_idx, src_channels[layer_idx], src_channels[layer_idx],self.cfg.backbone.pred_aspect_ratios[idx], self.cfg.backbone.pred_scales[idx],parent
            #0 0 256 256 [[1, 0.5, 2]] [24]  None
            #1 1 256 256 [[1, 0.5, 2]] [48]  PredictionModule
            #2 2 256 256 [[1, 0.5, 2]] [96]  PredictionModule
            #3 3 256 256 [[1, 0.5, 2]] [192] PredictionModule
            #4 4 256 256 [[1, 0.5, 2]] [384] PredictionModule
            #
            print(idx, layer_idx, src_channels[layer_idx], src_channels[layer_idx],pred_aspect_ratios[idx], pred_scales[idx],parent)
            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = pred_aspect_ratios[idx],
                                    scales        = pred_scales[idx],
                                    parent        = parent)
            self.prediction_layers.append(pred)
            #PredictionModule(
            #    (upfeature): Sequential(
            #        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #    (1): ReLU(inplace)
            #    )
            #    (bbox_layer): Conv2d(256, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #    (conf_layer): Conv2d(256, 243, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #    (mask_layer): Conv2d(256, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #    )

        # Extra parameters for the extra losses
        if self.use_class_existence_loss:
            # This comes from the smallest layer selected
            # Also note that self.cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], self.num_classes - 1)

        if self.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], self.num_classes-1, kernel_size=1)

        # For use in evaluation
        self.detect = Detect(self.num_classes, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)
        self.to(self.device)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def forward(self, batched_inputs): #images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        #if self.training and targets is None:
        #    raise ValueError("In training mode, targets should be passed")

        """ 
        batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
        Each item in the list contains the inputs for one image.
        For now, each item in the list is a dict that contains:

        * image: Tensor, image in (C, H, W) format.
        * instances (optional): groundtruth :class:`Instances`
        * proposals (optional): :class:`Instances`, precomputed proposals.

        Other information that's included in the original dicts, such as:

        * "height", "width" (int): the output resolution of the model, used in inference.
            See :meth:`postprocess` for details. 
        """
        if not self.training:
            return self.inference(batched_inputs)
        '''
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            targets = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            targets = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            targets = None
        
        #features = self.backbone(images.tensor)
        '''
        '''
        ### 1)
        backbone = build_backbone(self.cfg)
        num_images = 2
        images_tensor = torch.rand(num_images, 20, 30)
        image_sizes = [(10, 10), (20, 30)]
        images = ImageList(images_tensor, image_sizes)

        ### 2)
        #tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
        #image_sizes (list[tuple[int, int]]): Each tuple is (h, w).
        images = ImageList()#to_image_list(images)
        outs = self.backbone(images.tensors)

        ### 3)
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        '''
        #features = self.backbone(images.tensor)
        #result = features
        '''
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        '''
        images = torch.unsqueeze(batched_inputs[0][0], 0)
        targets = batched_inputs[1][0][0]
        masks = torch.unsqueeze(batched_inputs[1][1][0], 0)
        num_crowds = batched_inputs[1][2][0]

        #images, (targets, masks, num_crowds) = batched_inputs
        outs = self.backbone(images)#.tensor) #images.tensor) #caution : use ".tensor" !!!!
        proto_out = None
        if self.mask_type == self.mask_type_lincomb and self.eval_mask_branch:
            with timer.env('proto'):
                proto_x = images if self.proto_src is None else outs[self.proto_src] ## TODO:CHECK DIFF x vs images

                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)

                proto_out = self.proto_net(proto_x)
                proto_out = self.mask_proto_prototype_activation(proto_out)

                if self.mask_proto_prototypes_as_features:
                    # Clone here because we don't want to permute this, though idk if contiguous makes this unnecessary
                    proto_downsampled = proto_out.clone()

                    if self.mask_proto_prototypes_as_features_no_grad:
                        proto_downsampled = proto_out.detach()

                # Move the features last so the multiplication is easy
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

                if self.mask_proto_bias:
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)

        with timer.env('pred_heads'):
            pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }

            if self.use_instance_coeff:
                pred_outs['inst'] = []

            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                pred_x = outs[idx]

                if self.mask_type == self.mask_type_lincomb and self.mask_proto_prototypes_as_features:
                    # Scale the prototypes down to the current prediction layer's size and add it as inputs
                    proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].size()[2:], mode='bilinear', align_corners=False)
                    pred_x = torch.cat([pred_x, proto_downsampled], dim=1)

                # A hack for the way dataparallel works
                if self.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                    pred_layer.parent = [self.prediction_layers[0]]
                '''
                print('----------------------- BEFORE --------------------------', idx)
                if idx == 0:
                    print('pred_layer: ', pred_layer.upfeature._modules['0']._parameters['weight'].device)

                print('module_layer_0: ',self.prediction_layers._modules['0']._modules['upfeature']._modules['0']._parameters['weight'].device)
                print('module_layer_1: ',self.prediction_layers._modules['1'].parent[0]._modules['upfeature']._modules['0']._parameters['weight'].device)
                print('module_layer_2: ',self.prediction_layers._modules['2'].parent[0]._modules['upfeature']._modules['0']._parameters['weight'].device)
                print('module_layer_3: ',self.prediction_layers._modules['3'].parent[0]._modules['upfeature']._modules['0']._parameters['weight'].device)
                print('module_layer_4: ',self.prediction_layers._modules['4'].parent[0]._modules['upfeature']._modules['0']._parameters['weight'].device)
                print('-------------------------------------------------')
                '''
                p = pred_layer(pred_x, idx)
                '''
                print('----------------------- AFTER  --------------------------', idx)
                if idx == 0:
                    print('pred_layer: ', pred_layer.upfeature._modules['0']._parameters['weight'].device)
                print('module_layer_0: ',self.prediction_layers._modules['0']._modules['upfeature']._modules['0']._parameters['weight'].device)
                print('module_layer_1: ',self.prediction_layers._modules['1'].parent[0]._modules['upfeature']._modules['0']._parameters['weight'].device)
                print('module_layer_2: ',self.prediction_layers._modules['2'].parent[0]._modules['upfeature']._modules['0']._parameters['weight'].device)
                print('module_layer_3: ',self.prediction_layers._modules['3'].parent[0]._modules['upfeature']._modules['0']._parameters['weight'].device)
                print('module_layer_4: ',self.prediction_layers._modules['4'].parent[0]._modules['upfeature']._modules['0']._parameters['weight'].device)
                print('-------------------------------------------------')
                '''
                for k, v in p.items():
                    pred_outs[k].append(v)

        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)

        if proto_out is not None:
            pred_outs['proto'] = proto_out

        if self.training:

            # For the extra loss functions
            if self.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))

            if self.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])

            losses = {}
            yolact_loss=criterion(pred_outs, targets, masks, num_crowds)########TODO loss + data loader (does it load images, targets, masks, num_crowds correctly?)
            losses.update(yolact_loss)
            return losses
            #return pred_outs, loss
        else:
            if self.use_sigmoid_focal_loss:
                # Note: even though conf[0] exists, this mode doesn't train it so don't use it
                pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
            elif self.use_objectness_score:
                # See focal_loss_sigmoid in multibox_loss.py for details
                objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                pred_outs['conf'][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
                pred_outs['conf'][:, :, 0 ] = 1 - objectness
            else:
                pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)

            return self.detect(pred_outs)

        # result

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

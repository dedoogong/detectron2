import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
import numpy as np
from itertools import product
from math import sqrt
from typing import List

# from data.config import cfg, mask_type
from detectron2.modeling.detector.layers import Detect
from detectron2.modeling.detector.layers.interpolate import InterpolateModule
from detectron2.modeling.detector.backbone import construct_backbone

import torch.backends.cudnn as cudnn
#from detectron2.utils import timer
from detectron2.utils.functions import MovingAverage
from detectron2.modeling.detector.utils import timer as timer_yolact

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn


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
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False,
                                              **layer_cfg[2])
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


class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """

    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None):
        super().__init__()

        self.num_classes = 81  # cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.mask_dim = 32  # cfg.mask_dim
        self.num_priors = 3  # sum(len(x) for x in aspect_ratios)
        self.parent = [parent]  # Don't include this in the state dict
        head_layer_params = {'kernel_size': 3, 'padding': 1}
        num_instance_coeffs = 64
        extra_layers = (0, 0, 0)
        mask_type = 1
        mask_type_direct = 0
        mask_type_lincomb = 1
        self.extra_head_net = [(256, 3, {'padding': 1})]
        use_prediction_module = False
        use_instance_coeff = False
        self.mask_type = mask_type_lincomb
        if 0:  # cfg.mask_proto_prototypes_as_features
            in_channels += self.mask_dim

        if parent is None:
            if self.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, self.extra_head_net)

            if use_prediction_module:
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, **head_layer_params)

            if use_instance_coeff:  # cfg.use_instance_coeff
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * num_instance_coeffs, **head_layer_params)

            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in extra_layers]

            if self.mask_type == mask_type_lincomb and 0:  # self.mask_proto_coeff_gate
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None

    def forward(self, x, idx):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]

        conv_h = x.size(2)
        conv_w = x.size(3)
        # print('before model device: ', src.upfeature._modules['0'].weight.device, 'src.scales: ', src.scales, idx)
        # print('before x.device:', x.device)
        # print('which gpu:', self.customgpuname)
        ##  why first forward of src is correct( model-input match) but from second input->cuda:1, model->cuda:0??? why changed?
        ## why first replica is used suddenly??
        x = src.upfeature(x)
        '''
        if self.extra_head_net is not None:
            if src.upfeature._modules['0'].weight.device == x.device:
                x = src.upfeature(x)
            else:
                print('before model device: ', src.upfeature._modules['0'].weight.device)
                print('before x.device:', x.device)
                x=x.to(src.upfeature._modules['0'].weight.device)
                #src=src.to(x.device)
                x = src.upfeature(x)
                print('after model device: ', src.upfeature._modules['0'].weight.device)
                print('after x.device:', x.device)
        '''
        if False:  # cfg.use_prediction_module:
            # The two branches of PM design (c)
            a = src.block(x)

            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)

            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        if True:  # cfg.eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if False:  # cfg.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)

        # See box_utils.decode for an explanation of this
        if False:  # cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if True:  # cfg.eval_mask_branch:
            if False:  # cfg.mask_type == mask_type.direct:
                mask = torch.sigmoid(mask)
            elif True:  # cfg.mask_type == mask_type.lincomb:
                mask = torch.tanh(mask)  # cfg.mask_proto_coeff_activation(mask)

                if False:  # cfg.mask_proto_coeff_gate:
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)

        priors = self.make_priors(conv_h, conv_w)

        preds = {'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors}

        if False:  # cfg.use_instance_coeff:
            preds['inst'] = inst

        return preds

    def make_priors(self, conv_h, conv_w):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        # print("============make priors start================")
        with timer_yolact.env('makepriors'):
            if self.last_conv_size != (conv_w, conv_h):
                prior_data = []

                # Iteration order is important (it has to sync up with the convout)
                for j, i in product(range(conv_h), range(conv_w)):
                    # +0.5 because priors are in center-size notation
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h

                    for scale, ars in zip(self.scales, self.aspect_ratios):
                        for ar in ars:
                            if True:  # not cfg.backbone.preapply_sqrt:
                                ar = sqrt(ar)

                            if True:  # cfg.backbone.use_pixel_scales:
                                w = scale * ar / 550  # cfg.max_size
                                h = scale / ar / 550  # cfg.max_size
                            else:
                                w = scale * ar / conv_w
                                h = scale / ar / conv_h

                            # This is for backward compatability with a bug where I made everything square by accident
                            if True:  # cfg.backbone.use_square_anchors:
                                h = w

                            prior_data += [x, y, w, h]

                self.priors = torch.Tensor(prior_data).view(-1, 4).cuda()
                self.last_conv_size = (conv_w, conv_h)
                # print('self.last_conv_size : ',self.last_conv_size )

        return self.priors
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import torch

from ..registry import NECKS
from ..utils import ConvModule
from ..utils import pool_conv

@NECKS.register_module
class FPXN(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels =256,
                 num_levels = 5,
                 out_channels = 256,
                 rescale_level=2,
                 rescale_ratio=4,
                 top_conv = False,
                 after_add = False,
                 conv_cfg=None,
                 norm_cfg=None):
        super(FPXN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.rescale_level = rescale_level
        self.rescale_ratio = rescale_ratio
        self.rescale_channel = in_channels // rescale_ratio
        self.assemble_channel = self.rescale_channel * self.num_levels
        self.top_conv = top_conv
        self.after_add = after_add

        assert 0 <= self.rescale_level < self.num_levels
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        if self.top_conv:
            self.conv3 = nn.ModuleList()

        for i in range(self.num_levels):
            conv1 = ConvModule(
                self.in_channels,
                self.rescale_channel,
                1,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg
            )
            conv2 = ConvModule(
                self.assemble_channel,
                self.out_channels,
                1,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg
            )
            self.conv1.append(conv1)
            self.conv2.append(conv2)
            if self.top_conv:
                t_conv = ConvModule(
                    self.out_channels,
                    self.out_channels,
                    1,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg
                )
                self.conv3.append(t_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = [self.conv1[i](inputs[i]) for i in range(self.num_levels)]
        gather_size = inputs[self.refine_level].size()[2:]
        feats = [F.adaptive_max_pool2d(feats[i], gather_size) if i<self.rescale_level
                 else F.interpolate(feats[i], gather_size)
                 for i in range(self.num_levels)]
        feats = torch.cat(feats, dim=1)
        feats = self.conv2(feats)
        if self.top_conv:
            if self.after_add:
                outs = [self.conv3[i](inputs[i] + F.interpolate(feats, inputs[i].size()[2:])) if i<=self.rescale_level
                        else self.conv3[i](inputs[i] + F.adaptive_max_pool2d(feats, inputs[i].size()[2:]))
                        for i in range(self.num_levels)]
            else:
                outs = [inputs[i] + F.interpolate(self.conv3[i](feats), inputs[i].size()[2:]) if i<=self.rescale_level
                        else inputs[i] + F.adaptive_max_pool2d(self.conv3[i](feats), inputs[i].size()[2:])
                        for i in range(self.num_levels)]
        else:
            outs = [inputs[i] + F.interpolate(feats, inputs[i].size()[2:]) if i<=self.rescale_level
                    else inputs[i] + F.adaptive_max_pool2d(feats, inputs[i].size()[2:])
                    for i in range(self.num_levels)]

        return tuple(outs)
        # for i in range(self.num_levels):
        #     if i < self.refine_level:
        #         gathered = F.adaptive_max_pool2d(
        #             inputs[i], output_size=gather_size)
        #     else:
        #         gathered = F.interpolate(
        #             inputs[i], size=gather_size, mode='nearest')
        #     feats.append(gathered)

        # bsf = sum(feats) / len(feats)
        #
        # # step 2: refine gathered features
        # if self.refine_type is not None:
        #     bsf = self.refine(bsf)
        #
        # # step 3: scatter refined features to multi-levels by a residual path
        # outs = []
        # for i in range(self.num_levels):
        #     out_size = inputs[i].size()[2:]
        #     if i < self.refine_level:
        #         residual = F.interpolate(bsf, size=out_size, mode='nearest')
        #     else:
        #         residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
        #     outs.append(residual + inputs[i])
        #
        # return tuple(outs)

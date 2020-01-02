import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import torch

from ..registry import NECKS
from ..utils import ConvModule
from ..utils import pool_conv

@NECKS.register_module
class FuseFPN(nn.Module):
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
                 final_combine = 'concat',
                 conv_cfg=None,
                 norm_cfg=None):
        super(FuseFPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.final_combine = final_combine

        mid_channels = self.in_channels // 2
        self.res = nn.Sequential(
            ConvModule(self.in_channels, self.in_channels, 3, 1, padding=1,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg),
            ConvModule(self.in_channels, mid_channels, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg),
            ConvModule(mid_channels, self.out_channels, 3, 1, padding=1,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        )
        self.combine = ConvModule(self.in_channels*2, self.in_channels, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        if self.final_combine == 'concat':
            self.conv3 = nn.ModuleList()

        for i in range(self.num_levels):
            conv1 = ConvModule(
                self.in_channels,
                self.in_channels,
                1,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg
            )
            conv2 = ConvModule(
                self.in_channels,
                self.out_channels,
                1,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg
            )
            self.conv1.append(conv1)
            self.conv2.append(conv2)
            if self.final_combine == 'concat':
                conv3 = ConvModule(self.in_channels*2,
                                   self.in_channels,
                                   1,
                                   1,
                                   conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg
                                   )
                self.conv3.append(conv3)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, feats, semantic_feats):
        assert len(feats) == self.num_levels
        semantic_feats = torch.cat([self.res(semantic_feats), semantic_feats], dim=1)
        semantic_feats = self.combine(semantic_feats)

        # step 1: gather multi-level features by resize and average
        semantic_feats = [self.conv1[i](semantic_feats) for i in range(self.num_levels)]
        if self.final_combine == 'concat':
            outs = [torch.cat([F.interpolate(semantic_feats[i], feats[i].size()[-2:], mode='bilinear'), feats[i]], dim=1)
                    for i in range(self.num_levels)]
            outs = [self.conv3[i](outs[i]) for i in range(self.num_levels)]
        else:
            outs = [F.interpolate(semantic_feats[i], feats[i].size()[-2:], mode='bilinear') + feats[i] for i in range(self.num_levels)]
        outs = [self.conv2[i](outs[i]) for i in range(self.num_levels)]

        return tuple(outs)


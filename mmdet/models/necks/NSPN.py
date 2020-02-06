import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
import torch

from mmdet.core import auto_fp16, force_fp32
from mmdet.models.registry import NECKS
from mmdet.models.utils import ConvModule


@NECKS.register_module
class NSemanticPyramidNeck(nn.Module):

    def __init__(self,
                 feature_channels=256,
                 num_convs=3,
                 num_levels = 5,
                 conv_out_channels=256,
                 conv_cfg=None,
                 norm_cfg=None):
        super(NSemanticPyramidNeck, self).__init__()
        self.num_convs = num_convs
        self.num_levels = num_levels
        self.conv_out_channels = conv_out_channels
        self.feature_channels = feature_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False


        # self.lateral_convs = nn.ModuleList()
        self.combine_convs = nn.ModuleList()
        # self.proto_convs = nn.ModuleList()
        for i in range(num_levels):
            self.combine_convs.append(
                nn.Sequential(
                ConvModule(self.feature_channels,
                           self.conv_out_channels,
                           3,
                           padding=1,
                           norm_cfg=norm_cfg,
                           conv_cfg=conv_cfg),
                ConvModule(self.conv_out_channels,
                           self.conv_out_channels,
                           3,
                           padding=1,
                           norm_cfg=norm_cfg,
                           conv_cfg=conv_cfg),
                ConvModule(self.conv_out_channels,
                           self.conv_out_channels,
                           3,
                           padding=1,
                           norm_cfg=norm_cfg,
                           conv_cfg=conv_cfg)))

    def init_weights(self):
        pass

    @auto_fp16()
    def forward(self, feats):
        assert len(feats) == self.num_levels
        # masks = F.interpolate(masks, feats[0].size()[-2:])
        feats = list(feats)
        for i in range(self.num_levels):
                # protos = self.proto_convs[i](masks)
            feats[i] = self.combine_convs[i](feats[i]) + feats[i]
        return tuple(feats)

        # combine_feature = sum(features) / len(features)
        #         # masks = self.lateral_conv(masks)
        #         # combine_feature = torch.cat([combine_feature, masks], dim=1)
        #         # combine_feature = self.combine_conv(combine_feature)
        #         # outs = []
        #         # for i in range(self.num_levels):
        #         #     out_size = feats[i].size()[2:]
        #         #     if i < self.combine_level:
        #         #         residual = F.interpolate(combine_feature, size=out_size, mode='nearest')
        #         #     else:
        #         #         residual = F.adaptive_max_pool2d(combine_feature, out_size)
        #         #     outs.append(residual + feats[i])
        #         # return feats


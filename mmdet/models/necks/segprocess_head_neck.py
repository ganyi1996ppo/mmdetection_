import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
import torch

from mmdet.core import auto_fp16, force_fp32
from mmdet.models.registry import NECKS
from mmdet.models.utils import ConvModule


@NECKS.register_module
class SemanticProcessNeck(nn.Module):

    def __init__(self,
                 num_convs=4,
                 feature_channels=256,
                 mask_channels=80,
                 combine_level = 2,
                 num_levels = 5,
                 conv_out_channels=256,
                 groups=True,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SemanticProcessNeck, self).__init__()
        self.num_convs = num_convs
        self.mask_channels = mask_channels
        self.combine_level = combine_level
        self.num_levels = num_levels
        self.conv_out_channels = conv_out_channels
        self.feature_channels = feature_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.groups = groups

        if self.groups:
            self.lateral_conv = ConvModule(self.mask_channels,self.mask_channels, 3, stride=2, padding=1,
                                           groups = self.mask_channels,
                                           conv_cfg=self.conv_cfg,
                                           norm_cfg=self.norm_cfg)
        else:
            self.lateral_conv = ConvModule(
                self.mask_channels, self.mask_channels, 3, stride=2, padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg
            )

        # self.convs = nn.ModuleList()
        # for i in range(self.num_convs):
        #     # in_channels = self.in_channels if i == 0 else conv_out_channels
        #     if self.groups:
        #         self.convs.append(
        #             ConvModule(
        #                 conv_out_channels,
        #                 conv_out_channels,
        #                 3,
        #                 padding=1,
        #                 groups=in_channels,
        #                 conv_cfg=self.conv_cfg,
        #                 norm_cfg=self.norm_cfg))
        #     else:
        #         self.convs.append(
        #             ConvModule(
        #                 conv_out_channels,
        #                 conv_out_channels,
        #                 3,
        #                 padding=1,
        #                 conv_cfg=self.conv_cfg,
        #                 norm_cfg=self.norm_cfg))

        # self.conv_embedding = ConvModule(
        #     conv_out_channels,
        #     conv_out_channels,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg)
        # self.conv_logits = nn.Conv2d(conv_out_channels, self.num_classes, 1)
        self.combine_conv = ConvModule(self.mask_channels+self.feature_channels,
                                       self.conv_out_channels,
                                       3,
                                       padding=1,
                                       norm_cfg=norm_cfg,
                                       conv_cfg=conv_cfg)

    def init_weights(self):
        pass

    @auto_fp16()
    def forward(self, feats, masks):
        assert len(feats) == self.num_levels

        features = []
        gather_size = feats[self.combine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.combine_level:
                gathered = F.interpolate(
                    feats[i], size=gather_size, mode='nearest')
            else:
                gathered = F.adaptive_max_pool2d(
                    feats[i], gather_size)
            features.append(gathered)

        combine_feature = sum(features) / len(features)
        masks = self.lateral_conv(masks)
        combine_feature = torch.cat([combine_feature, masks], dim=1)
        combine_feature = self.combine_conv(combine_feature)
        outs = []
        for i in range(self.num_levels):
            out_size = feats[i].size()[2:]
            if i < self.combine_level:
                residual = F.interpolate(combine_feature, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(combine_feature, out_size)
            outs.append(residual + feats[i])
        return feats


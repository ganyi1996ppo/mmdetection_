import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
import torch

from mmdet.core import auto_fp16, force_fp32
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class RelationHead(nn.Module):
    """Multi-level fused semantic segmentation head.

    in_1 -> 1x1 conv ---
                        |
    in_2 -> 1x1 conv -- |
                       ||
    in_3 -> 1x1 conv - ||
                      |||                  /-> 1x1 conv (mask prediction)
    in_4 -> 1x1 conv -----> 3x3 convs (*4)
                        |                  \-> 1x1 conv (feature)
    in_5 -> 1x1 conv ---
    """  # noqa: W605

    def __init__(self,
                 num_convs=4,
                 in_channels=183,
                 conv_out_channels=256,
                 num_classes=183,
                 conv_cfg=None,
                 norm_cfg=None):
        super(RelationHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.encode_conv = ConvModule(in_channels, conv_out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_convs):
            self.lateral_convs.append(
                ConvModule(
                    self.conv_out_channels,
                    self.conv_out_channels,
                    3,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False))


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, feats):
        feats = torch.softmax(feats, dim=1)
        feats = self.encode_conv(feats)
        for conv in self.lateral_convs:
            feats = conv(feats)

        return feats

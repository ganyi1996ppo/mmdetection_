import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init

from mmdet.core import auto_fp16, force_fp32
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class SemanticProcessHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 in_channels=80,
                 conv_out_channels=256,
                 groups=True,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SemanticProcessHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.groups = groups

        if self.groups:
            self.lateral_conv = ConvModule(self.in_channels,self.conv_out_channels, 3, stride=2, padding=1,
                                           groups = self.in_channels,
                                           conv_cfg=self.conv_cfg,
                                           norm_cfg=self.norm_cfg)
        else:
            self.lateral_conv = ConvModule(
                self.in_channels, self.conv_out_channels, 3, stride=2, padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg
            )

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else conv_out_channels
            if self.groups:
                self.convs.append(
                    ConvModule(
                        in_channels,
                        conv_out_channels,
                        3,
                        padding=1,
                        groups=in_channels,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            else:
                self.convs.append(
                    ConvModule(
                        in_channels,
                        conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))

        # self.conv_embedding = ConvModule(
        #     conv_out_channels,
        #     conv_out_channels,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg)
        # self.conv_logits = nn.Conv2d(conv_out_channels, self.num_classes, 1)


    def init_weights(self):
        pass

    @auto_fp16()
    def forward(self, feats):
        feats = self.lateral_conv(feats)
        for conv in self.convs:
            feats = conv(feats)
        return feats


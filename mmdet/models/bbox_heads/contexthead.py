import torch.nn as nn
import torch
import numpy as np
import mmcv
import cv2

from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class ContextHead(nn.Module):
    def __int__(self,
                pooling = 'avg',
                in_channels = 256,
                out_channels = 256,
                num_classes = 81,
                num_convs = 3,
                conv_cfg=None,
                norm_cfg=None):
        super(ContextHead, self).__init__()
        if pooling == 'max':
            self.pooling = nn.MaxPool2d(2, 2)
        else:
            self.pooling = nn.AvgPool2d(2,2)
        self.num_convs = num_convs
        self.num_classes=  num_classes
        self.Convs = nn.ModuleList()
        for i in range(num_convs):
            in_feature = num_classes+in_channels if i==0 else in_channels
            conv = ConvModule(
                in_feature,
                out_channels,
                1,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg
            )
            self.Convs.append(conv)
        self.init_weight()

    def forward(self, x, mask_pred):
        mask_pred = self.pooling(mask_pred)
        x = torch.cat([x, mask_pred], dim=1)
        for conv in self.Convs:
            x = conv(x)
        return x





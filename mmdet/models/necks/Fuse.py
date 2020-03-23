import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import NECKS
from mmcv.cnn import xavier_init
from ..utils import ConvModule


@NECKS.register_module
class FuseNeck(nn.Module):

    def __init___(self,
                  in_channels=256,
                  out_channels=256,
                  num_levels=5,
                  final_combine='add',
                  norm_cfg = None,
                  conv_cfg=None
                  ):
        super(FuseNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.final_combine = final_combine
        self.combine_convs = nn.ModuleList()
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        for i in range(self.num_levels):
            conv = ConvModule(in_channels, out_channels, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg)
            self.combine_convs.append(conv)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, old_feas, feats):
        assert len(old_feas) == self.num_levels, "old feats have wrong level"
        assert len(feats) == self.num_levels, "new feats have wrong level"

        for i in range(self.num_levels):
            feats[i]  = self.combine_convs[i](old_feas[i] + feats[i])

        return feats



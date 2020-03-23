import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import NECKS
from mmcv.cnn import xavier_init
from ..utils import ConvModule

@NECKS.register_module
class GAS(nn.Module):

    def __init__(self,
                 gather_level=2,
                 in_channels=256,
                 out_channels=256,
                 num_levels = 5,
                 final_combine = 'add',
                 conv_before=False,
                 conv_cfg=None,
                 norm_cfg=None):
        super(GAS, self).__init__()
        self.gather_conv = nn.ModuleList()
        if self.conv_before:
            self.bf_conv = nn.ModuleList()
        self.gather_level = gather_level
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert final_combine in ['add', 'con'], "unimplemented"
        self.conv_before = conv_before
        self.final_conbine = final_combine
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_levels = num_levels
        for i in range(num_levels):
            ex_conv = ConvModule(self.in_channels, self.out_channels, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
            self.gather_conv.append(ex_conv)
            if conv_before:
                bf_conv = ConvModule(self.in_channels, self.out_channels, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
                self.bf_conv.append(bf_conv)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, feats):
        assert len(feats) == self.num_levels, "feature number don't match"
        H,W = feats[self.gather_level].size()[2:]
        new_feats = []
        for i in range(self.num_levels):
            if i == self.gather_level:
                new_feats.append(feats[i])
            else:
                new_feats.append(F.interpolate(feats[i],(H,W), mode='bilinear'))

        if self.conv_before:
            for i in range(len(new_feats)):
                new_feats[i] = self.bf_conv[i](new_feats[i])
        gather_feat = sum(new_feats)
        for i in range(len(new_feats)):
            new_feats[i] = F.interpolate(self.gather_conv[i](gather_feat), feats[i].size()[2:], mode='bilinear')

        return feats, new_feats


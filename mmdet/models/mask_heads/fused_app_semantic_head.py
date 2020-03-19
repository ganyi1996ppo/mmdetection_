import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import kaiming_init

from mmdet.core import auto_fp16, force_fp32
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class FusedAPPSemanticHead(nn.Module):
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
                 num_ins,
                 fusion_level,
                 num_convs=4,
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=183,
                 ignore_label=255,
                 loss_weight=0.2,
                 app_size = (1,2,3,6),
                 conv_cfg=None,
                 norm_cfg=None):
        super(FusedAPPSemanticHead, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.app_size = app_size
        self.fp16_enabled = False

        self.stages = nn.ModuleList()
        for i in range(len(app_size)):
            self.stages.append(nn.Conv2d(self.in_channels, self.in_channels//4, 1, bias=False))

        self.bottleneck = ConvModule(self.in_channels*2, self.in_channels, 1)
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False))


        self.psp_module = nn.ModuleList()
        for i in range(len(app_size)):
            self.psp_module.append(
                self.make_stage(in_channels, (self.app_size[i], self.app_size[i]), in_channels//len(app_size))
            )


        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else conv_out_channels
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
        self.conv_logits = nn.Conv2d(conv_out_channels*2, self.num_classes, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def init_weights(self):
        kaiming_init(self.conv_logits)
        for module in self.stages:
                if isinstance(module, nn.Conv2d):
                    kaiming_init(module)

    def make_stage(self, in_feature, pooling_size, out_feature):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(pooling_size),
            ConvModule(in_feature, out_feature, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        )

    # def add_stage(self, size):
    #     return nn.Sequential(
    #         nn.AdaptiveAvgPool2d(output_size=(size, size)),
    #         nn.Conv2d(self.in_channels, self.in_channels//4, 1, bias=False)
    #     )

    @auto_fp16()
    def forward(self, feats):
        x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                x += self.lateral_convs[i](feat)

        for i in range(self.num_convs):
            x = self.convs[i](x)

        out = [x]
        for i, psp in enumerate(self.psp_module):
            temp = psp(x)
            out.append(F.interpolate(temp, x.size()[-2:], mode='bilinear'))
        x = torch.cat(out, dim=1)


        # h, w = feats[-1].size()[-2:]
        # above_feats = [self.stages[i](F.adaptive_avg_pool2d(feats[-1], self.app_size[i])) for i in range(len(self.app_size))]
        # above_feats = [F.interpolate(feat, (h,w), mode='bilinear') for feat in above_feats] + [feats[-1]]
        # feats[-1] = self.bottleneck(torch.cat(above_feats, dim=1))
        # x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])
        # fused_size = tuple(x.shape[-2:])
        # for i, feat in enumerate(feats):
        #     if i != self.fusion_level:
        #         feat = F.interpolate(
        #             feat, size=fused_size, mode='bilinear', align_corners=True)
        #         x += self.lateral_convs[i](feat)
        #
        # for i in range(self.num_convs):
        #     x = self.convs[i](x)
        #
        # mask_pred = self.conv_logits(x)
        x = self.conv_logits(x)
        return  x

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, labels):
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg *= self.loss_weight
        return loss_semantic_seg

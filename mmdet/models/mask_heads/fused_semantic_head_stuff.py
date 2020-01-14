import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
from ..builder import build_loss
import mmcv
import numpy as np
import torch

from mmdet.core import auto_fp16, force_fp32, mask_target
from ..registry import HEADS
from ..utils import ConvModule


@HEADS.register_module
class FusedMaskHead(nn.Module):
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
                 proto_out=32,
                 ignore_label=255,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss',use_mask=True, loss_weight=1.0
                 )):
        super(FusedMaskHead, self).__init__()
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.proto_out = proto_out
        self.ignore_label = ignore_label
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_mask = build_loss(loss_mask)
        self.fp16_enabled = False

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
        self.conv_logits = nn.Conv2d(conv_out_channels, self.proto_out, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)


    def init_weights(self):
        kaiming_init(self.conv_logits)

    def get_target(self, sampling_results, gt_masks, proto_cfg):
        proposals = [res.bboxes for res in sampling_results]
        assgined_gt_inds = [
            res.inds for res in sampling_results
        ]
        all_target = mask_target(proposals, assgined_gt_inds, gt_masks, proto_cfg)
        return all_target

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

        mask_pred = self.conv_logits(x)
        # x = self.conv_embedding(x)
        return  mask_pred

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, labels):
        # labels = labels.squeeze(1).long()
        # H,W = mask_pred.size()[-2:]
        # labels = [mmcv.imresize(label, (W,H), interpolation='nearest') for label in labels]
        # labels = np.stack(labels)
        # labels = torch.from_numpy(labels).to(mask_pred.device).float()
        losses = dict()
        loss_mask = self.loss_mask(mask_pred, labels, torch.zeros(mask_pred.size(0)).long())
        losses['proto_mask'] = loss_mask
        return losses

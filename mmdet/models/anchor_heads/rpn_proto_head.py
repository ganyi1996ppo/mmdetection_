import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from ..utils import ConvModule
from ..registry import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module
class RPNProtoHead(AnchorHead):

    def __init__(self,
                 in_channels,
                 share_conv=3,
                 num_proto=32,
                 **kwargs):
        self.share_conv = share_conv
        self.num_proto = num_proto
        super(RPNProtoHead, self).__init__(2, in_channels, **kwargs)


    def _init_layers(self):
        self.rpn_conv = nn.ModuleList()
        for i in range(self.share_conv):
            in_channels = self.in_channels if i==0 else self.feat_channels
            Conv = ConvModule(in_channels,self.feat_channels, 3, padding=1)
            self.rpn_conv.append(Conv)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.rpn_coef = nn.Conv2d(self.feat_channels, self.num_anchors * self.num_proto, 1)

    def init_weights(self):
        normal_init(self.rpn_coef, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        for conv in self.rpn_conv:
            x = conv(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        rpn_coef = self.rpn_coef(x)
        return rpn_cls_score, rpn_bbox_pred, rpn_coef

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(RPNProtoHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def get_bboxes(self, cls_scores, bbox_preds, rpn_coef, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        coeffs = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            coeff_list = [
                rpn_coef[i][img_id] for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals, coeff = self.get_bboxes_single(cls_score_list, bbox_pred_list, coeff_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
            coeffs.append(coeff)
        return result_list, coeffs

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          coeff_pred,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        mlvl_coeff = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            rpn_coeffs_pred = coeff_pred[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            rpn_coeffs_pred = rpn_coeffs_pred.permute(1, 2, 0).reshape(-1, self.num_proto)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
                rpn_coeffs_pred = rpn_coeffs_pred[topk_inds, :]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                rpn_coeffs_pred = rpn_coeffs_pred[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, proposals_ind = nms(proposals, cfg.nms_thr)
            rpn_coeffs_pred = rpn_coeffs_pred[proposals_ind, :]
            proposals = proposals[:cfg.nms_post, :]
            rpn_coeffs_pred = rpn_coeffs_pred[:cfg.nms_post,:]
            mlvl_proposals.append(proposals)
            mlvl_coeff.append(rpn_coeffs_pred)
        proposals = torch.cat(mlvl_proposals, 0)
        coeffs = torch.cat(mlvl_coeff, 0)
        if cfg.nms_across_levels:
            proposals, idx = nms(proposals, cfg.nms_thr)
            coeffs = coeffs[idx, :]
            proposals = proposals[:cfg.max_num, :]
            coeffs = coeffs[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
            coeffs = coeffs[topk_inds, :]
        return proposals, coeffs

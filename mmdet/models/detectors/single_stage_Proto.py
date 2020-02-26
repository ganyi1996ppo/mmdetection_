import torch.nn as nn
import torch

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class Retina_Proto(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 semantic_head=None,
                 fuse_neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Retina_Proto, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.semantic_head = builder.build_head(semantic_head)
        self.fuse_neck = builder.build_neck(fuse_neck)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        self.semantic_head.init_weights()
        self.fuse_neck.init_weights()

    def init_weights(self, pretrained=None):
        super(Retina_Proto, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        losses = dict()
        semantic_pred = self.semantic_head(x)
        loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
        semantic_pred = semantic_pred.detach()
        losses['loss_mask_seg'] = loss_seg
        seg_inds = torch.cat([torch.arange(1, 12), torch.arange(13, 26), torch.arange(27, 29), torch.arange(31, 45),
                              torch.arange(46, 66), torch.arange(67, 68), torch.arange(70, 71),
                              torch.arange(72, 83), torch.arange(84, 91)])
        seg_feats = semantic_pred.softmax(dim=1)
        seg_feats = seg_feats[:, seg_inds, :,:].contiguous()
        x = self.fuse_neck(x, seg_feats)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        loss = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(loss)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        semantic_pred = self.semantic_head(x)
        seg_inds = torch.cat([torch.arange(1, 12), torch.arange(13, 26), torch.arange(27, 29), torch.arange(31, 45),
                              torch.arange(46, 66), torch.arange(67, 68), torch.arange(70, 71),
                              torch.arange(72, 83), torch.arange(84, 91)])
        seg_feats = semantic_pred.softmax(dim=1)
        seg_feats = seg_feats[:, seg_inds, :, :].contiguous()
        x = self.fuse_neck(x, seg_feats)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

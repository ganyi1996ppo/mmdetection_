import torch
import  torch.nn.functional as F

from mmdet.core import bbox2roi, build_assigner, build_sampler, bbox2result
from .. import builder
from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class ProtoRCNN(TwoStageDetector):
    """Mask Scoring RCNN.

    https://arxiv.org/abs/1903.00241
    """
    #TODO: add semantic backgorund predict into the scope to improve the map
    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 semantic_head,
                 fuse_neck=None,
                 semantic_roi_extractor=None,
                 mask_relation_head=None,
                 rpn_proto=False,
                 proto_mask_training=True,
                 proto_combine = 'sum',
                 neck=None,
                 shared_head=None,
                 pretrained=None):

        self.rpn_proto = rpn_proto
        self.proto_mask = proto_mask_training
        self.proto_combine = proto_combine
        super(ProtoRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.semantic_head = builder.build_head(semantic_head)
        self.semantic_roi_extractor=False

        self.augneck = False
        self.relation_head=False
        if mask_relation_head:
            self.relation_head = True
            self.mask_relation_head = builder.build_head(mask_relation_head)
        if fuse_neck:
            self.fuse_neck = builder.build_neck(fuse_neck)
            self.augneck = True
            self.fuse_neck.init_weights()
        self.semantic_head.init_weights()
        # self.with_bg = with_bg
        if semantic_roi_extractor:
            self.semantic_roi_extractor = builder.build_roi_extractor(semantic_roi_extractor)
            self.semantic_extract = True

    def forward_dummy(self, img):
        raise NotImplementedError

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    # TODO: refactor forward_train in two stage to reduce code redundancy
    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()
        semantic_pred= self.semantic_head(x)
        # loss_seg = self.semantic_head.loss(semantic_pred, fg_masks)
        # losses['loss_mask_seg'] = loss_seg
        if self.augneck:
            x = self.fuse_neck(x)

        # RPN forward and loss
        if self.with_rpn:
            if self.rpn_proto:
                rpn_cls, rpn_bbox, rpn_proto = self.rpn_head([torch.cat([feat, F.interpolate(semantic_pred, feat.size()[-2:])], 1) for feat in x])
            else:
                rpn_cls, rpn_bbox, rpn_proto = self.rpn_head(x)
            rpn_loss_inputs = (rpn_cls, rpn_bbox) + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = (rpn_cls, rpn_bbox, rpn_proto) + (img_meta, proposal_cfg)
            proposal_list, coeffs = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            coeff_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                coeff = torch.cat([coeffs[i][assign_result.max_overlaps], coeffs[i]],dim=0)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
                coeff_results.append(coeff[torch.cat([sampling_result.pos_inds, sampling_result.neg_inds],dim=0)])
                # proto_coeffs.append(coeff_list[i][sampling_result.pos_inds])

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            coeffs = torch.cat(coeff_results, 0)
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            if self.semantic_extract:
                    # and self.relation_head:
                # relations = self.mask_relation_head(semantic_pred)
                seg_feats = self.semantic_roi_extractor([semantic_pred], rois)
                if self.proto_combine == 'sum':
                    seg_feats = (seg_feats * coeffs[:,:,None,None]).sum(dim=1,keepdim=True)
                    if self.proto_mask:
                        if self.train_cfg.proto.using_clamp:
                            seg_feats = seg_feats.clamp(0,1)
                        else:
                            seg_feats = (seg_feats>self.train_cfg.proto.mask_thr).float()
                        proto_targets = self.semantic_head.get_target(sampling_results, gt_masks, self.train_cfg.proto)
                        loss_proto = self.semantic_head.loss(seg_feats, proto_targets)
                        losses.update(loss_proto)
                        seg_feats = seg_feats.detach()

                elif self.proto_combine == 'con':
                    seg_feats = seg_feats * coeffs[:,:,None,None]
                    if self.proto_mask:
                        raise NotImplementedError

                # _, sem_feats = torch.max(semantic_pred, dim=1)
                # sem_feats = sem_feats[:,None,:,:]
                # sem_feats = torch.zeros(sem_feats.size(0), 183, sem_feats.size(2),
                #                         sem_feats.size(3)).to(sem_feats.device).scatter_(1,sem_feats,1)
                # fg_feats = sem_feats[:, 1:91, :, :].contiguous()
                # fg_feats = self.semantic_roi_extractor([fg_feats],rois)
                # _, inds = torch.sum(fg_feats, (2, 3)).max(dim=1)
                # fg_feats = fg_feats[torch.arange(fg_feats.size(0)),inds,:,:]
                # fg_feats = fg_feats[:,None,:,:]
                cls_score, bbox_pred = self.bbox_head(bbox_feats, seg_feats)

            else:
                cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

            # mask iou head forward and loss
            # pos_mask_pred = mask_pred[range(mask_pred.size(0)), pos_labels]
            # mask_iou_pred = self.mask_iou_head(mask_feats, pos_mask_pred)
            # pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)
            #                                         ), pos_labels]
            # mask_iou_targets = self.mask_iou_head.get_target(
            #     sampling_results, gt_masks, pos_mask_pred, mask_targets,
            #     self.train_cfg.rcnn)
            # loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred,
            #                                         mask_iou_targets)
            # losses.update(loss_mask_iou)
        return losses

    def simple_seg_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           seg_feats,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head(roi_feats, seg_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def simple_test_rpn(self, x, img_meta, rpn_test_cfg):
        rpn_out = self.rpn_head(x)
        proposal_inputs = rpn_out + (img_meta, rpn_test_cfg)
        proposal_list, params = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list, params

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        semantic_pred = self.semantic_head(x)
        if self.augneck:
            x = self.fuse_neck(x)
        if self.rpn_proto:
            proposal_list, params = self.simple_test_rpn(
                [torch.cat([feat, F.interpolate(semantic_pred, feat.size()[-2:])], 1) for feat in x], img_meta, self.test_cfg.rpn if proposals is None else proposals
            )
        else:
            proposal_list, params = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        if self.semantic_extract:
            if self.semantic_extract:
                rois = bbox2roi(proposal_list)
                params = torch.cat(params,0)
                # relation_feats = self.mask_relation_head(semantic_pred)
                proto_rois = self.semantic_roi_extractor([semantic_pred], rois)
                if self.proto_combine == 'sum':
                    proto_rois = (proto_rois*params[:,:,None,None]).sum(dim=1, keepdim=True)
                    if self.test_cfg.proto.using_clamp:
                        proto_rois = proto_rois.clamp(0,1)
                    else:
                        proto_rois = (proto_rois>self.train_cfg.proto.mask_thr).float()
                elif self.proto_combine == 'con':
                    proto_rois = (proto_rois*params[:,:,None,None])
                # _, sem_feats = torch.max(semantic_pred, dim=1)
                # sem_feats = sem_feats[:, None, :, :]
                # sem_feats = torch.zeros(sem_feats.size(0), 183, sem_feats.size(2),
                #                         sem_feats.size(3)).to(sem_feats.device).scatter_(1, sem_feats, 1)
                # fg_feats = sem_feats[:, 1:91, :, :].contiguous()
                # fg_feats = self.semantic_roi_extractor([fg_feats], rois)
                # _, inds = torch.sum(fg_feats, (2, 3)).max(dim=1)
                # fg_feats = fg_feats[torch.arange(fg_feats.size(0)), inds, :, :]
                # fg_feats = fg_feats[:, None, :, :]
            # semantic_rois = self.semantic_roi_extractor(semantic_pred[:self.semantic_roi_extractor.num_inputs], rois)
            # _, sem_feats = torch.max(semantic_rois, dim=1)
            # sem_feats = sem_feats[:,None,:,:]
            # sem_feats = torch.zeros(sem_feats.size(0), 183, sem_feats.size(2),
            #                         sem_feats.size(3)).to(sem_feats.device).scatter_(1, sem_feats, 1)
            # fg_feats = sem_feats[:, 1:91, :, :].contiguous()
                det_bboxes, det_labels = self.simple_seg_test_bboxes(
                x, img_meta, proposal_list, proto_rois, self.test_cfg.rcnn, rescale=rescale)
        else:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

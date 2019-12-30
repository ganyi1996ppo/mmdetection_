import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as  F

from mmdet.core import auto_fp16, force_fp32, mask_target, mask_bg_target
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule
from ..losses import accuracy


@HEADS.register_module
class FCNMaskHead_back(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs=2,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 # fc_out_channels=1024,      #
                 # mask_score_thr=0.05,       #
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 # loss_cls=dict(
                 #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                 ):
        super(FCNMaskHead_back, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        # self.fc_out_channels = fc_out_channels          #
        # self.mask_score_thr = mask_score_thr            #
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)
        # self.loss_cls = build_loss(loss_cls)            #

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

        # classification layers
        # for i in range(self.num_fcs):
        #     fc_in_channels = self.conv_out_channels if i==0 else self.fc_out_channels
        #     self.fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        # self.cls_fc = nn.Linear(self.fc_out_channels, num_classes)

        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.debug_imgs = None
        self.transform1 = ConvModule(81, 64, 1, padding = 0, conv_cfg = self.conv_cfg, norm_cfg = self.norm_cfg)
        self.transform2 = ConvModule(64, 32, 3, stride=2, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.transform3 = ConvModule(32, 16, 3, stride=2, padding=1, conv_cfg=self.conv_cfg, norm_cfg=norm_cfg)
        self.transform4 = ConvModule(16, 8, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)
        self.transform5 = ConvModule(8, 1, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        refine = self.transform1(mask_pred)
        refine = self.transform2(refine)
        refine = self.transform3(refine)
        refine = self.transform4(refine)
        refine = self.transform5(refine)
        # x = self.cls_pooling(x)
        # x = x.view(x.size(0), -1)
        # for fc in self.fcs:
        #     x = self.relu(fc(x))
        # x_cls = self.cls_fc(x)
        return mask_pred, refine#, x_cls

    # def fetch_mask(self, mask_pred, mask_cls):
    #     assert mask_pred.size(0) == mask_cls.size(0)
    #     if mask_pred.size(0):
    #         mask_inds = mask_cls.argmax(dim=1)
    #         batch_size = mask_cls.size(0)
    #         batch_inds = torch.arange(batch_size)
    #         fetched_mask = mask_pred[batch_inds, mask_inds]
    #         fetched_mask = fetched_mask.unsqueeze(1)
    #
    #     else:
    #         fetched_mask = mask_pred.new_full((1024,1,28,28), 1)
    #
    #     return fetched_mask


    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        # pos_proposals = [res.pos_bboxes for res in sampling_results]
        # pos_assigned_gt_inds = [
        #     res.pos_assigned_gt_inds for res in sampling_results
        # ]
        # mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
        #                            gt_masks, rcnn_train_cfg)
        proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def get_bg_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        proposals = [res.pos_bboxes for res in sampling_results]
        mask_bg_targets = mask_bg_target(proposals, gt_masks, rcnn_train_cfg)
        return mask_bg_targets

    def get_all_target(self, sampling_results, gt_mask, rcnn_train_cfg):
        proposals = [res.bboxes for res in sampling_results]
        assigned_gt_inds = [
            res.inds for res in sampling_results
        ]
        all_target = mask_target(proposals, assigned_gt_inds,
                                 gt_mask, rcnn_train_cfg)

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_refine, mask_targets, labels):
        loss = dict()
        # if len(_mask_pred.size())!=4:
        #     _mask_pred.unsqueeze(0)
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        if len(mask_targets.size())==3:
            mask_targets = mask_targets[:,None,:,:]
        H,W = mask_refine.size()[-2:]
        mask_targets = F.interpolate(mask_targets, (H,W)).squeeze()
            # loss_cls = self.loss_cls(mask_cls_pred, labels)
        loss_refine = self.loss_mask(mask_refine, mask_targets, torch.zeros_like(labels))

        # loss['loss_mask_cls'] = loss_cls
        loss['loss_mask'] = loss_mask
        loss['lossa_refine'] = loss_refine
        # loss['accuracy_mask_cls'] = accuracy(mask_cls_pred, labels)
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms

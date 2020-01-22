import torch.nn as nn

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead
import torch
import torch.nn.functional as F
import mmcv
from mmdet.core import mask_target, mask_bg_target, force_fp32, bbox_target, bbox_overlaps
from ..losses import accuracy
from ..builder import build_loss


@HEADS.register_module
class ConvFCBBoxHead_MH(BBoxHead):
    """More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 mask_channels=256,
                 using_mask = True,
                 with_IoU = False,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 proto_combine='con',
                 feature_reduce=False,
                 # mask_conv=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 using_bg=False,
                 using_refine=True,
                 loss_iou = dict(type='MSELoss', loss_weight=0.5),
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead_MH, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.using_mask = using_mask
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.using_bg = using_bg
        self.using_refine = using_refine
        self.with_IoU = with_IoU
        self.mask_channels = mask_channels
        self.proto_combine = proto_combine
        self.feature_reduce = feature_reduce
        if with_IoU:
            self.iou_loss = build_loss(loss_iou)

        # self.hint_conv = ConvModule(self.mask_channels, self.mask_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)

        # add shared convs and fcs
        if self.proto_combine == 'None':
            if self.feature_reduce:
                self.reduce_con = ConvModule(self.in_channels, conv_out_channels - mask_channels, 1, conv_cfg=conv_cfg,
                                             norm_cfg=norm_cfg)
        else:
            combine_channels = self.in_channels + self.mask_channels if proto_combine == 'con' else self.in_channels
            self.combine = ConvModule(combine_channels, conv_out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        # self.mask_conv = nn.ModuleList()
        # for i in range(mask_conv):
        #     conv_m = ConvModule(1, 1, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
        #     self.mask_conv.append(conv_m)

        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
        if self.with_IoU:
            self.IoU_reg = nn.Linear(self.reg_last_dim, self.num_classes)


    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead_MH, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    # @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    # def loss(self,
    #          cls_score,
    #          bbox_pred,
    #          labels,
    #          label_weights,
    #          bbox_targets,
    #          bbox_weights,
    #          reduction_override=None):
    #     losses = dict()
    #     if cls_score is not None:
    #         avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
    #         losses['loss_cls_refine'] = self.loss_cls(
    #             cls_score,
    #             labels,
    #             label_weights,
    #             avg_factor=avg_factor,
    #             reduction_override=reduction_override)
    #         losses['acc_refine'] = accuracy(cls_score, labels)
    #     if bbox_pred is not None:
    #         pos_inds = labels > 0
    #         if self.reg_class_agnostic:
    #             pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
    #         else:
    #             pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
    #                                            4)[pos_inds, labels[pos_inds]]
    #         losses['loss_bbox_refine'] = self.loss_bbox(
    #             pos_bbox_pred,
    #             bbox_targets[pos_inds],
    #             bbox_weights[pos_inds],
    #             avg_factor=bbox_targets.size(0),
    #             reduction_override=reduction_override)
    #     return losses

    #TODO: add IoU target aquire and loss calculation
    def get_iou_target(self, sampling_reuslt, bbox_pred, bbox_target):
        pos_proposals = [res.pos_bboxes for res in sampling_reuslt]
        pos_assigned_gt_inds = [
            res.pos_gt_assigned_gt_inds for res in sampling_reuslt
        ]
        # bbox_overlaps()




    def get_mask_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        # pos_proposals = [res.pos_bboxes for res in sampling_results]
        # pos_assigned_gt_inds = [
        #     res.pos_assigned_gt_inds for res in sampling_results
        # ]
        # mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
        #                            gt_masks, rcnn_train_cfg)
        proposals = [res.bboxes for res in sampling_results]
        assigned_gt_inds = [
            res.inds for res in sampling_results
        ]
        mask_targets = mask_target(proposals, assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        mask_bg_targets = mask_bg_target(proposals, gt_masks, rcnn_train_cfg)
        return mask_targets, mask_bg_targets

    # def get_target(self, sampling_results, gt_bboxes, gt_labels,
    #                rcnn_train_cfg):
    #     pos_proposals = [res.pos_bboxes for res in sampling_results]
    #     neg_proposals = [torch.tensor([]) for res in sampling_results]
    #     pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
    #     pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
    #     reg_classes = 1 if self.reg_class_agnostic else self.num_classes
    #     cls_reg_targets = bbox_target(
    #         pos_proposals,
    #         neg_proposals,
    #         pos_gt_bboxes,
    #         pos_gt_labels,
    #         rcnn_train_cfg,
    #         reg_classes,
    #         target_means=self.target_means,
    #         target_stds=self.target_stds)
    #     return cls_reg_targets


    def forward(self, x, mask_pred):
        # shared part
        if self.using_mask:
            # for conv in self.mask_conv:
            #     mask_pred = conv(mask_pred)
            # mask_pred = self.hint_conv(mask_pred)

            if self.proto_combine == 'con':
                x = torch.cat([x, mask_pred], dim=1)
                x = self.combine(x)
            elif self.proto_combine == 'sum':
                x = x + mask_pred
                x = self.combine(x)
            else:
                x = self.reduce_con(x)
                x = torch.cat([x, mask_pred], dim=1)
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.with_IoU:
            IoU_pred = self.IoU_reg(x_reg)
            return cls_score, bbox_pred, IoU_pred
        return cls_score, bbox_pred


@HEADS.register_module
class SharedFCBBoxHead_MH(ConvFCBBoxHead_MH):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead_MH, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

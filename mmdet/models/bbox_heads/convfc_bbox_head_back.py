import torch.nn as nn
import torch
import numpy as np
import mmcv
import cv2

from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead


@HEADS.register_module
class ConvFCBBoxHead_back(BBoxHead):
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
                 # mask_out_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 max_pooling = True,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead_back, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        # self.mask_out_channels = mask_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # self.mask_convs = ConvModule(self.num_classes,
        #                              mask_out_channels,
        #                              1,
        #                              conv_cfg=self.conv_cfg,
        #                              norm_cfg=self.norm_cfg)
        self.max_pooling = max_pooling
        if self.max_pooling:
            self.pooling = nn.AdaptiveMaxPool2d(self.roi_feat_size)
        else:
            self.pooling = nn.AdaptiveAvgPool2d(self.roi_feat_size)

        # add shared convs and fcs
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
        self.mask_convs = nn.ModuleList()
        for i in range(3):
            mask_conv_dim = 81 if i<2 else 1
            self.mask_convs.append(ConvModule(81, mask_conv_dim, 1, padding=0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
        # self.bbox_mask_conv = ConvModule(256,self.num_classes,1,conv_cfg=conv_cfg,norm_cfg=norm_cfg)


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
        super(ConvFCBBoxHead_back, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)


    def ShowMidlleResult(self, mask, det_bbox, img_meta, rcnn_test_cfg):
        if isinstance(mask, torch.Tensor):
            mask =mask.sigmoid().cpu().numpy()
        assert isinstance(mask, np.ndarray)
        mask = mask.astype(np.float32)
        bboxes = det_bbox.cpu().numpy()[:,:4]
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        img = img_meta[0]['img']
        img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
        img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
        multi_bboxes = bboxes
        masks = mask
        for i in range(det_bbox.size()[0]):
            img_show = img
            mask = masks[i]
            bboxes = multi_bboxes[i]
            bbox_h = max(bboxes[2] - bboxes[0] + 1, 1)
            bbox_w = max(bboxes[3] - bboxes[1] + 1, 1)
            im_mask = mmcv.imresize(mask, (bbox_w, bbox_h))
            im_mask = (im_mask > rcnn_test_cfg.mask_thr_binary).astype(np.uint8)
            cv2.rectangle(img_show, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), color=[255, 0, 0])
            img_show[bboxes[0]:bboxes[0]+bbox_h, bboxes[2]:bboxes[2] + bbox_w] *= 0.5
            img_show[bboxes[0]:bboxes[0] + bbox_h, bboxes[2]:bboxes[2] + bbox_w] += 0.5 * im_mask
            cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('image', img_show)
            key = cv2.waitKey(0)
            if (key == 's'):
                cv2.imwrite('./image' + str(i), img_show)
            elif (key == 'b'):
                break
            else:
                continue

    def forward(self, x, mask_pred):

        #2019/10/25
        for mask_conv in self.mask_convs:
            mask_pred = mask_conv(mask_pred)

        mask_pred = self.pooling(mask_pred)

        #unknow when
        # mask_feats = self.mask_convs(mask_feats)
        # mask_pred = mask_pred.unsqueeze(1)
        x = torch.cat([x, mask_pred], dim=1)
        # shared part

        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        # x = self.bbox_mask_conv(x)
        # zip_x = zip(x.chunk(81,dim=1), mask_pred.chunk(81, dim=1))
        # x = torch.cat([torch.cat(elem, dim=1) for elem in zip_x], dim=1)

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
        return cls_score, bbox_pred


@HEADS.register_module
class SharedFCBBoxHead_back(ConvFCBBoxHead_back):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCBBoxHead_back, self).__init__(
            num_shared_convs=3,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

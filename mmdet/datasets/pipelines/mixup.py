import os.path as osp
import warnings

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.utils.data._utils.collate

from ..registry import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile_mixup(object):

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        if isinstance(results, list):
            res = []
            for result in results:
                filename = osp.join(result['img_prefix'],
                                    result['img_info']['filename'])
                img = mmcv.imread(filename)
                if self.to_float32:
                    img = img.astype(np.float32)
                result['filename'] = filename
                result['img'] = img
                result['img_shape'] = img.shape
                result['ori_shape'] = img.shape
                res.append(result)
            return res
        else:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
            img = mmcv.imread(filename)
            if self.to_float32:
                img = img.astype(np.float32)
            results['filename'] = filename
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(to_float32={})'.format(
            self.to_float32)


@PIPELINES.register_module
class LoadAnnotations_mixup(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_mask_box=False,               # Change
                 with_seg=False,
                 poly2mask=True,
                 skip_img_without_anno=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_mask_box = with_mask_box          # Change
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.skip_img_without_anno = skip_img_without_anno

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']
        if len(results['gt_bboxes']) == 0 and self.skip_img_without_anno:
            file_path = osp.join(results['img_prefix'],
                                 results['img_info']['filename'])
            warnings.warn(
                'Skip the image "{}" that has no valid gt bbox'.format(
                    file_path))
            return None
        results['gt_bboxes_ignore'] = ann_info.get('bboxes_ignore', None)
        results['bbox_fields'].extend(['gt_bboxes', 'gt_bboxes_ignore'])
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        gt_masks.append(np.zeros_like(gt_masks[0]))
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results


    def _load_mask_boxes(self, results):
        """
        Change
        :param results:
        :return:
        """
        gt_mask = results['gt_masks']
        x_list = [], y_list = []
        for i in range(len(gt_mask)):
            if i%2 == 0:
                x_list.append(int(gt_mask[i]))
            else:
                y_list.append(int(gt_mask[i]))
        point_list = zip(x_list, y_list)
        mb_xmin = min(x_list)
        mb_ymin = min(y_list)
        mb_xmax = max(x_list)
        mb_ymax = max(y_list)
        results['mask_boxes'] = (mb_xmin, mb_ymin, mb_xmax, mb_ymax)
        return results


    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        return results

    def __call__(self, results):
        if isinstance(results, list):
            res = []
            for result in results:
                if self.with_bbox:
                    results = self._load_bboxes(result)
                    if results is None:
                        return None
                if self.with_label:
                    results = self._load_labels(result)
                if self.with_mask:
                    results = self._load_masks(result)
                if self.with_mask_box:
                    results = self._load_mask_boxes(result)
                if self.with_seg:
                    results = self._load_semantic_seg(result)
                res.append(results)
            return res
        else:
            if self.with_bbox:
                results = self._load_bboxes(results)
                if results is None:
                    return None
            if self.with_label:
                results = self._load_labels(results)
            if self.with_mask:
                results = self._load_masks(results)
            if self.with_mask_box:
                results = self._load_mask_boxes(results)
            if self.with_seg:
                results = self._load_semantic_seg(results)
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str


@PIPELINES.register_module
class MixUp(object):

    def __init__(self, alpha, beta):
        self.alpha= alpha
        self.beta = beta

    def __call__(self, results):
        result1, result2 = results
        H = max(result1['image_shape'][0], result2['image_shape'][0])
        W = max(result1['image_shape'][1], result2['image_shape'][1])
        image_shape = (H,W,3)
        image = np.zeros(image_shape)
        image[0:result1['image_shape'][0], 0:result1['image_shape'][1],:] += result1['img'] * self.alpha
        image[0:result2['image_shape'][0], 0:result2['image_shape'][1],:] += result2['img'] * self.beta
        bboxes1, bboxes2 = result1['gt_bboxes'],result2['gt_bboxes']
        label1, label2 = result1['gt_labels'], result2['gt_labels']
        bboxes = np.vstack([bboxes1, bboxes2])
        labels = np.vstack([label1, label2])
        bboxes_loss = np.vstack([np.full((bboxes1.shape[0],1), self.alpha), np.full((bboxes2.shape[0],1), self.beta)])
        results['img'] = image
        results['gt_bboxes'] = bboxes
        results['gt_labels'] = labels
        results['bboxes_loss'] = bboxes_loss
        results['bbox_fields'].extend(['gt_bboxes'])
        results['filename'] = result1['filename']
        results['ori_shape'] = image.shape
        results['img_shape'] = image.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(alpha={}, beta={})'.format(self.alpha, self.beta))
        return repr_str





# @PIPELINES.register_module
# class LoadProposals(object):
#
#     def __init__(self, num_max_proposals=None):
#         self.num_max_proposals = num_max_proposals
#
#     def __call__(self, results):
#         proposals = results['proposals']
#         if proposals.shape[1] not in (4, 5):
#             raise AssertionError(
#                 'proposals should have shapes (n, 4) or (n, 5), '
#                 'but found {}'.format(proposals.shape))
#         proposals = proposals[:, :4]
#
#         if self.num_max_proposals is not None:
#             proposals = proposals[:self.num_max_proposals]
#
#         if len(proposals) == 0:
#             proposals = np.array([0, 0, 0, 0], dtype=np.float32)
#         results['proposals'] = proposals
#         results['bbox_fields'].append('proposals')
#         return results
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(num_max_proposals={})'.format(
#             self.num_max_proposals)

import mmcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair

def mask_bg_target(proposal_list, gt_mask_list, cfg):
    cfg_list = [cfg for _ in range(len(proposal_list))]
    mask_bg_targets = map(mask_bg_target_single, proposal_list, gt_mask_list, cfg_list)
    mask_bg_targets = torch.cat(list(mask_bg_targets))
    return mask_bg_targets

def mask_bg_target_single(proposals, gt_masks, cfg):
    mask_size = _pair(cfg.mask_size)
    num = proposals.size(0)
    mask_bg_targets = []
    if num > 0:
        proposals_np = proposals.cpu().numpy()
        for i in range(num):
            gt_bg_mask = gt_masks[-1]
            bbox = proposals_np[i, :].astype(np.int32)
            x1,y1,x2,y2 = bbox
            w = np.maximum(x2-x1+1, 1)
            h = np.maximum(y2-y1+1, 1)
            target = mmcv.imresize(gt_bg_mask[y1:y1+h, x1:x1+w], mask_size[::-1])
            mask_bg_targets.append(target)
        mask_bg_targets = torch.from_numpy(np.stack(mask_bg_targets)).float().to(proposals.device)
    else:
        mask_bg_targets = proposals.new_zeros((0. ) + mask_size)

    return mask_bg_targets


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            # mask_size (h, w) to (w, h)
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   mask_size[::-1])
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
    return mask_targets

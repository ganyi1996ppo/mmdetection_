import torch
import numpy as np


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


def mask_overlaps(pos_proposals, pos_assigned_gt_inds, gt_masks):
    """Compute area ratio of the gt mask inside the proposal and the gt
    mask of the corresponding instance"""
    if len(pos_proposals.size()) == 1:
        pos_proposals = pos_proposals[None, :]
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        area_ratios = []
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        if len(proposals_np.shape) == 1:
            proposals_np = proposals_np[None, :]
        # compute mask areas of gt instances (batch processing for speedup)
        gt_instance_mask_area = np.array([gt_mask.sum((-1, -2)) for gt_mask in gt_masks])
        for i in range(num_pos):
            ind = pos_assigned_gt_inds[i]
            gt_mask = gt_masks[ind]

            # crop the gt mask inside the proposal
            x1, y1, x2, y2 = proposals_np[i, :].astype(np.int32)
            gt_mask_in_proposal = gt_mask[y1:y2 + 1, x1:x2 + 1]

            ratio = gt_mask_in_proposal.sum() / (
                    gt_instance_mask_area[ind] + 1e-7)
            area_ratios.append(ratio)
        area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(
            pos_proposals.device)
    else:
        area_ratios = pos_proposals.new_zeros((0,))
    return area_ratios
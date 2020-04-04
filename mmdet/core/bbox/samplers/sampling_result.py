import torch


class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.diluted_neg_inds = assign_result.gt_inds[torch.cat([pos_inds, neg_inds])] == -2
        self.diluted_weight = assign_result.max_overlaps[self.diluted_neg_inds]
        self.diluted_labels = assign_result.labels[self.diluted_neg_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.neg_gt_inds = assign_result.anchor_overlaps[neg_inds - self.num_gts]
        # self.inds = torch.cat([self.pos_assigned_gt_inds, self.neg_gt_inds])
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
            self.neg_gt_labels = assign_result.labels[neg_inds]
        else:
            self.pos_gt_labels = None



    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])
    @property
    def labels(self):
        return torch.cat([self.pos_gt_labels, self.neg_gt_labels])

    @property
    def inds(self):
        return torch.cat([self.pos_assigned_gt_inds, self.neg_gt_inds])


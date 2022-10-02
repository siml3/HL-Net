# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, cls_agnostic_bbox_reg=False):
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def assign_label_to_proposals(self, proposals, targets):
        for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            match_quality_matrix = boxlist_iou(target, proposal)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # Fast RCNN only need "labels" field for selecting the targets
            target = target.copy_with_fields(["labels", "attributes"])
            matched_targets = target[matched_idxs.clamp(min=0)]
            
            labels_per_image = matched_targets.get_field("labels").to(dtype=torch.int64)
            attris_per_image = matched_targets.get_field("attributes").to(dtype=torch.int64)

            labels_per_image[matched_idxs < 0] = 0
            attris_per_image[matched_idxs < 0, :] = 0
            proposals[img_idx].add_field("labels", labels_per_image)
            proposals[img_idx].add_field("attributes", attris_per_image)
        return proposals


    def __call__(self, class_logits, box_regression, proposals):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])
            proposals (list[BoxList])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat([proposal.get_field("regression_targets") for proposal in proposals], dim=0)

        classification_loss = F.cross_entropy(class_logits, labels.long())

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


class EMB_FastRNNLossComputation(FastRCNNLossComputation):
    def __init__(self, cls_agnostic_bbox_reg=False):
        super(EMB_FastRNNLossComputation, self).__init__(cls_agnostic_bbox_reg)

    def __call__(self, emd_preds, ref_preds, proposals):
        cls_logit_emd0, cls_logit_emd1, bbox_pred_emd0, bbox_pred_emd1 = emd_preds
        cls_logit_ref0, cls_logit_ref1, bbox_pred_ref0, bbox_pred_ref1 = ref_preds

        loss0 = embed_loss_softmax(
            bbox_pred_emd0, cls_logit_emd0, bbox_pred_emd1, 
            cls_logit_emd1, proposals, self.cls_agnostic_bbox_reg,
        )
        loss1 = embed_loss_softmax(
            bbox_pred_emd1, cls_logit_emd1, bbox_pred_emd0, 
            cls_logit_emd0, proposals, self.cls_agnostic_bbox_reg
        )
        loss2 = embed_loss_softmax(
            bbox_pred_ref0, cls_logit_ref0, bbox_pred_ref1, 
            cls_logit_ref1, proposals, self.cls_agnostic_bbox_reg
        )
        loss3 = embed_loss_softmax(
            bbox_pred_ref1, cls_logit_ref1, bbox_pred_ref0, 
            cls_logit_ref0, proposals, self.cls_agnostic_bbox_reg
        )

        # loss_rcnn = torch.cat([loss0, loss1], dim=1)
        # loss_ref = torch.cat([loss2, loss3], dim=1)
        # # requires_grad = False
        # _, min_indices_rcnn = loss_rcnn.min(dim=1)
        # _, min_indices_ref = loss_ref.min(dim=1)

        # loss_rcnn = loss_rcnn.gather(1, min_indices_rcnn.unsqueeze(1)).mean()
        # loss_ref = loss_ref.gather(1, min_indices_ref.unsqueeze(1)).mean()

        loss_rcnn = torch.where(loss0 <= loss1, loss0, loss1).mean()
        loss_ref = torch.where(loss2 <= loss3, loss2, loss3).mean()

        return loss_rcnn, loss_ref


def smooth_l1_loss_noreduce(input, target, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss

# @staticmethod
def embed_loss_softmax(p_b0, p_s0, p_b1, p_s1, proposals, cls_agnostic_bbox_reg):
    device = p_b0.device
    pred_delta = torch.cat([p_b0, p_b1], dim=1).view(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], dim=1).view(-1, p_s0.shape[-1])
    labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0).contiguous().view(-1)
    regression_targets = cat([proposal.get_field("regression_targets") for proposal in proposals], dim=0)
    regression_targets = regression_targets.contiguous().view(-1, regression_targets.shape[-1])

    rm_ignore_mask = (labels >= 0).to(pred_score)
    labels = labels.clamp(min=0)


    # assert labels.min() >= 0
    # assert pred_score.shape[0] ==  labels.shape[0]
    assert labels.max() < pred_score.shape[-1]
    classification_loss = F.cross_entropy(pred_score, labels.long(), reduction='none') * rm_ignore_mask

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).view(-1)
    labels_pos = labels[sampled_pos_inds_subset]
    if cls_agnostic_bbox_reg:
        map_inds = torch.tensor([4, 5, 6, 7], device=device)
    else:
        map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

    # if len(map_inds) > 0:
    #     assert map_inds.max() < pred_delta.shape[-1]
    # assert labels.shape[0] == pred_delta.shape[0]
    assert labels.shape[0] == regression_targets.shape[0]

    box_loss = smooth_l1_loss_noreduce(
        pred_delta[sampled_pos_inds_subset[:, None], map_inds],
        regression_targets[sampled_pos_inds_subset],
        beta=1,
    )

    # box_loss = smooth_l1_loss_noreduce(
    #     pred_delta,
    #     regression_targets.repeat(1, pred_delta.shape[-1] // regression_targets.shape[-1]),
    #     beta=1,
    # )

    loss_container = torch.zeros_like(classification_loss)
    # assert labels.shape[0] == loss_container.shape[0]
    loss_container[sampled_pos_inds_subset] = loss_container[sampled_pos_inds_subset] + box_loss.sum(1)
    loss_container = loss_container + classification_loss

    # loss_container = classification_loss + box_loss.sum(1)

    loss = loss_container.view(-1, 2).sum(1, keepdim=True)

    # classification_loss[sampled_pos_inds_subset] = classification_loss[sampled_pos_inds_subset] + box_loss.sum(1)

    # loss = classification_loss.view(-1, 2).sum(1)

    return loss


def make_roi_box_loss_evaluator(cfg):
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(cls_agnostic_bbox_reg)

    return loss_evaluator

def make_roi_box_loss_evaluator_emd(cfg):
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = EMB_FastRNNLossComputation(cls_agnostic_bbox_reg)

    return loss_evaluator

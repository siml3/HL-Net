# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import torch.nn.functional as F

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor, make_roi_box_post_processor_emd
from .loss import make_roi_box_loss_evaluator, make_roi_box_loss_evaluator_emd
from .sampling import make_roi_box_samp_processor, make_roi_box_samp_processor_emd
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_union
import ipdb

def add_predict_logits(proposals, class_logits):
    slice_idxs = [0]
    for i in range(len(proposals)):
        slice_idxs.append(len(proposals[i])+slice_idxs[-1])
        proposals[i].add_field("predict_logits", class_logits[slice_idxs[i]:slice_idxs[i+1]])
    return proposals

def add_predict_logits_emd(proposals, class_logits0, class_logits1):
    slice_idxs = [0]
    class_logits = torch.stack((class_logits0, class_logits1), dim=1)
    for i in range(len(proposals)):
        slice_idxs.append(len(proposals[i])+slice_idxs[-1])
        proposals[i].add_field("predict_logits", class_logits[slice_idxs[i]:slice_idxs[i+1]])
        # proposals[i].add_field("predict_logits0", class_logits0[slice_idxs[i]:slice_idxs[i+1]])
        # proposals[i].add_field("predict_logits1", class_logits1[slice_idxs[i]:slice_idxs[i+1]])
    return proposals



@registry.ROI_BOX_HEAD.register("Default")
class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=self.cfg.MODEL.ATTRIBUTE_ON)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.samp_processor = make_roi_box_samp_processor(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        ###################################################################
        # box head specifically for relation prediction model
        ###################################################################
        if self.cfg.MODEL.RELATION_ON:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                # use ground truth box as proposals
                # try:
                #     proposals = [target.copy_with_fields(["labels", "attributes", "attention_map", "gamma"]) for target in targets]
                # except:
                #     try:
                #         proposals = [target.copy_with_fields(["labels", "attributes", "gamma"]) for target in targets]
                #     except:
                #         proposals = [target.copy_with_fields(["labels", "attributes"]) for target in targets]
                proposals = [target.copy_with_fields(["labels", "attributes", "gamma"]) for target in targets]
                '''
                TODO: we must add gamma features to sgdet as well!
                '''
                # ipdb.set_trace()
                x = self.feature_extractor(features, proposals)
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                    # mode==predcls
                    # return gt proposals and no loss even during training
                    return x, proposals, {}
                else:
                    # mode==sgcls
                    # add field:class_logits into gt proposals, note field:labels is still gt
                    class_logits, _ = self.predictor(x)
                    proposals = add_predict_logits(proposals, class_logits)
                    return x, proposals, {}
            else:
                # mode==sgdet
                if self.training or not self.cfg.TEST.CUSTUM_EVAL:
                    proposals = self.samp_processor.assign_label_to_proposals(proposals, targets)
                x = self.feature_extractor(features, proposals)
                class_logits, box_regression = self.predictor(x)
                proposals = add_predict_logits(proposals, class_logits)
                # post process:
                # filter proposals using nms, keep original bbox, add a field 'boxes_per_cls' of size (#nms, #cls, 4)
                x, result = self.post_processor((x, class_logits, box_regression), proposals, relation_mode=True)
                # note x is not matched with processed_proposals, so sharing x is not permitted
                return x, result, {}

        #####################################################################
        # Original box head (relation_on = False)
        #####################################################################
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.samp_processor.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        
        if not self.training:
            x, result = self.post_processor((x, class_logits, box_regression), proposals)

            # if we want to save the proposals, we need sort them by confidence first.
            if self.cfg.TEST.SAVE_PROPOSALS:
                _, sort_ind = result.get_field("pred_scores").view(-1).sort(dim=0, descending=True)
                x = x[sort_ind]
                result = result[sort_ind]
                result.add_field("features", x.cpu().numpy())

            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], proposals)

        return x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

@registry.ROI_BOX_HEAD.register("EMD")
class ROIBoxHead_EMD(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead_EMD, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=self.cfg.MODEL.ATTRIBUTE_ON)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor_emd(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator_emd(cfg)
        self.samp_processor = make_roi_box_samp_processor_emd(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        ###################################################################
        # box head specifically for relation prediction model
        ###################################################################
        if self.cfg.MODEL.RELATION_ON:
            # TODO: must reconstruct the code before run relation.
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                # use ground truth box as proposals
                proposals = [target.copy_with_fields(["labels", "attributes"]) for target in targets]
                x = self.feature_extractor(features, proposals)
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                    # mode==predcls
                    # return gt proposals and no loss even during training
                    return x, proposals, {}
                else:
                    # mode==sgcls
                    # add field:class_logits into gt proposals, note field:labels is still gt
                    # TODO: use emd predict 
                    cls_logit_emd0, cls_logit_emd1, bbox_pred_emd0, bbox_pred_emd1, \
                    cls_logit_ref0, cls_logit_ref1, bbox_pred_ref0, bbox_pred_ref1 = self.predictor(x)
                    # class_logits = (cls_logit_ref0 + cls_logit_ref1) * 0.5
                    # class_logits = torch.max(cls_logit_ref0, cls_logit_ref1)
                    proposals = add_predict_logits_emd(proposals, cls_logit_ref0, cls_logit_ref1)
                    # ipdb.set_trace()
                    return x, proposals, {}
            else:
                # mode==sgdet

                x = self.feature_extractor(features, proposals)
                cls_logit_emd0, cls_logit_emd1, bbox_pred_emd0, bbox_pred_emd1, \
                cls_logit_ref0, cls_logit_ref1, bbox_pred_ref0, bbox_pred_ref1 = self.predictor(x)
                # class_logits = (cls_logit_ref0 + cls_logit_ref1) * 0.5
                # class_logits = torch.max(cls_logit_ref0, cls_logit_ref1)
                proposals = add_predict_logits_emd(proposals, cls_logit_ref0, cls_logit_ref1)
                # post process:
                # filter proposals using nms, keep original bbox, add a field 'boxes_per_cls' of size (#nms, #cls, 4)
                x, result = self.post_processor((x, cls_logit_ref0, cls_logit_ref1, 
                            bbox_pred_ref0, bbox_pred_ref1), 
                            proposals, relation_mode=True)
                # note x is not matched with processed_proposals, so sharing x is not permitted
                if self.training or not self.cfg.TEST.CUSTUM_EVAL:
                    result = self.samp_processor.assign_label_to_proposals(result, targets)
                for det_boxes in result:
                    det_boxes.extra_fields['labels'] = det_boxes.extra_fields['labels'][:, 0]
                    det_boxes.extra_fields['attributes'] = det_boxes.extra_fields['attributes'][:, 0]
                return x, result, {}

        #####################################################################
        # Original box head (relation_on = False)
        #####################################################################
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.samp_processor.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        cls_logit_emd0, cls_logit_emd1, bbox_pred_emd0, bbox_pred_emd1, \
        cls_logit_ref0, cls_logit_ref1, bbox_pred_ref0, bbox_pred_ref1 = self.predictor(x)
        
        if not self.training:
            x, result = self.post_processor((x, cls_logit_ref0, cls_logit_ref1, 
                        bbox_pred_ref0, bbox_pred_ref1), 
                        proposals)

            # if we want to save the proposals, we need sort them by confidence first.
            if self.cfg.TEST.SAVE_PROPOSALS:
                _, sort_ind = result.get_field("pred_scores").view(-1).sort(dim=0, descending=True)
                # if len(sort_ind) > 0:
                #     assert sort_ind.max() < x.shape[0]
                x = x[sort_ind]
                result = result[sort_ind]
                result.add_field("features", x.cpu().numpy())

            return x, result, {}

        # loss emd, loss_ref
        loss_emd, loss_ref = self.loss_evaluator([cls_logit_emd0, cls_logit_emd1, 
            bbox_pred_emd0, bbox_pred_emd1], [cls_logit_ref0, cls_logit_ref1, bbox_pred_ref0, 
            bbox_pred_ref1], proposals)

        return x, proposals, dict(loss_emd=loss_emd, loss_ref=loss_ref)

    


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    func = registry.ROI_BOX_HEAD[cfg.MODEL.ROI_BOX_HEAD.META_ARCH]
    return func(cfg, in_channels)

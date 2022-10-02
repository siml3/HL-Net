# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn
import torch


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor_EMD")
class FPNPredictor_EMD(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor_EMD, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.emd0_cls_score = nn.Linear(representation_size, num_classes)
        self.emd1_cls_score = nn.Linear(representation_size, num_classes)
        self.ref0_cls_score = nn.Linear(representation_size, num_classes)
        self.ref1_cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.emd0_bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.emd1_bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.ref0_bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.ref1_bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.emd0_cls_score.weight, std=0.01)
        nn.init.normal_(self.emd1_cls_score.weight, std=0.01)
        nn.init.normal_(self.ref0_cls_score.weight, std=0.01)
        nn.init.normal_(self.ref1_cls_score.weight, std=0.01)
        nn.init.normal_(self.emd0_bbox_pred.weight, std=0.001)
        nn.init.normal_(self.emd1_bbox_pred.weight, std=0.001)
        nn.init.normal_(self.ref0_bbox_pred.weight, std=0.001)
        nn.init.normal_(self.ref1_bbox_pred.weight, std=0.001)

        self.fc = nn.Linear(representation_size+(num_classes-1)*5*4, representation_size)
        nn.init.kaiming_uniform_(self.fc.weight, a=1)
        nn.init.constant_(self.fc.bias, 0)
        self.relu = nn.ReLU()

        for l in [self.emd0_cls_score, self.emd1_cls_score, self.emd0_bbox_pred, self.emd1_bbox_pred]:
            nn.init.constant_(l.bias, 0)



    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        cls_logit_emd0 = self.emd0_cls_score(x)
        bbox_pred_emd0 = self.emd0_bbox_pred(x)
        cls_logit_emd1 = self.emd1_cls_score(x)
        bbox_pred_emd1 = self.emd1_bbox_pred(x)

        boxes_feature_0 = torch.cat((bbox_pred_emd0[:, 4:], cls_logit_emd0[:, 1:]), dim=1).repeat(1, 4)
        boxes_feature_1 = torch.cat((bbox_pred_emd1[:, 4:], cls_logit_emd1[:, 1:]), dim=1).repeat(1, 4)

        boxes_feature_0 = torch.cat((x, boxes_feature_0), dim=1)
        boxes_feature_1 = torch.cat((x, boxes_feature_1), dim=1)

        refine_x0 = self.relu(self.fc(boxes_feature_0))
        refine_x1 = self.relu(self.fc(boxes_feature_1))

        cls_logit_ref0 = self.ref0_cls_score(refine_x0)
        bbox_pred_ref0 = self.ref0_bbox_pred(refine_x0)

        cls_logit_ref1 = self.ref1_cls_score(refine_x1)
        bbox_pred_ref1 = self.ref1_bbox_pred(refine_x1)

        return (cls_logit_emd0, cls_logit_emd1, bbox_pred_emd0, bbox_pred_emd1, 
                cls_logit_ref0, cls_logit_ref1, bbox_pred_ref0, bbox_pred_ref1)


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)

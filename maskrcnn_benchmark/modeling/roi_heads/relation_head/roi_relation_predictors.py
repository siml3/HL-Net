# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_union, cat_boxlist_nofield
from maskrcnn_benchmark.modeling.utils import cat
from .model_motifs import FrequencyBias
from .utils_relation import layer_init
from maskrcnn_benchmark.data import get_dataset_statistics
from .model_gpsnet import Boxes_Encode
from .model_mp import Boxes_Encode, ARTs
from math import pi
from scipy.stats import entropy


@registry.ROI_RELATION_PREDICTOR.register("HLNetPredictor")
class HLNetPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(HLNetPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.context_layer = ARTs(config, obj_classes, self.pooling_dim,
                                           200, self.hidden_dim)
        self.get_boxes_encode = Boxes_Encode()

        self.ort_embedding = nn.Parameter(self.get_ort_embeds(self.num_obj_cls, 200), requires_grad=False)

        self.post_emb_s = nn.Linear(self.pooling_dim, self.pooling_dim)
        layer_init(self.post_emb_s, xavier=True)
        # self.post_emb_s.weight = torch.nn.init.xavier_normal(self.post_emb_s.weight, gain=1.0)
        self.post_emb_o = nn.Linear(self.pooling_dim, self.pooling_dim)
        layer_init(self.post_emb_o, xavier=True)
        # self.post_emb_o.weight = torch.nn.init.xavier_normal(self.post_emb_o.weight, gain=1.0)
        self.rel_compress = nn.Linear(self.pooling_dim + 64, self.num_rel_cls, bias=True)
        layer_init(self.rel_compress, xavier=True)
        # self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)

        # self.freq_gate.weight = torch.nn.init.xavier_normal(self.freq_gate.weight, gain=1.0)

        self.merge_obj_high = nn.Linear(self.hidden_dim, self.pooling_dim,)
        layer_init(self.merge_obj_high, xavier=True)
        self.merge_obj_low = nn.Linear(self.pooling_dim + 5 + 200, self.pooling_dim)
        layer_init(self.merge_obj_low, xavier=True)

        self.hmp_ws = nn.Linear(self.pooling_dim, self.hidden_dim)
        self.hmp_wo = nn.Linear(self.pooling_dim, self.hidden_dim)
        self.hmp_wu = nn.Linear(self.pooling_dim, self.hidden_dim)
        self.rel_hmp = nn.Linear(self.hidden_dim, 1)
        self.tanh = nn.Tanh()

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_gate = nn.Linear(self.pooling_dim + 64, self.num_rel_cls, bias=True)
            layer_init(self.freq_gate, xavier=True)
            self.freq_bias = FrequencyBias(config, statistics)


    @staticmethod
    def get_ort_embeds(k, dims):
        ind = torch.arange(1, k+1).float().unsqueeze(1).repeat(1,dims)
        lin_space = torch.linspace(-pi, pi, dims).unsqueeze(0).repeat(k,1)
        t = ind * lin_space
        return torch.sin(t) + torch.cos(t)

    @staticmethod
    def intersect_2d_tensor(x1, x2):
        if x1.size(1) != x2.size(1):
            raise ValueError("Input arrays must have same #columns")

        res = (x1[..., None] == x2.t()[None, ...]).prod(1)
        return res

    def get_hmp_infos(self, proposals, pair_idxs, rel_labels_full, obj_feats, union_features, features, union_features_extractor):
        head_feat_list = []
        tail_feat_list = []
        head_bbox_list = []
        tail_bbox_list = []
        hmp_pair_idxs_list = []
        hmp_labels_lists = []
        unique_inverse_indices_list = []
        unique_indices_list = []

        for proposal, pair_idx, rel_label_full, obj_feat, union_feat in zip(
            proposals, pair_idxs, rel_labels_full, obj_feats, union_features):
            if len(pair_idx) == 0:
                head_bbox_list.append(None)
                tail_bbox_list.append(None)
                head_feat_list.append(torch.empty((0, obj_feat.shape[-1])).to(obj_feat))
                tail_feat_list.append(torch.empty((0, obj_feat.shape[-1])).to(obj_feat))
                hmp_labels_lists.append(torch.empty((0,)).to(obj_feat))
                hmp_pair_idxs_list.append(torch.empty((0,2)).to(pair_idx))
                continue
            iou = boxlist_iou(proposal, proposal)
            iou = iou * (1-torch.eye(len(iou))).to(iou)
            mask_sub_same_for_sign = (pair_idx[:,0][:,None]==pair_idx[:,0][None,:])&(iou[pair_idx[:,1]][:,pair_idx[:,1]] >= 0.5)
            mask_obj_same_for_sign = (pair_idx[:,1][:,None]==pair_idx[:,1][None,:])&(iou[pair_idx[:,0]][:,pair_idx[:,0]] >= 0.5)

            sub_same_sign_pair_idx = torch.nonzero(mask_sub_same_for_sign)
            obj_same_sign_pair_idx = torch.nonzero(mask_obj_same_for_sign)

            hmp_pair_idxs_list.append(torch.cat([sub_same_sign_pair_idx, obj_same_sign_pair_idx], dim=0))
            sign_pair_idxs = torch.cat([sub_same_sign_pair_idx, obj_same_sign_pair_idx], dim=0)
            if len(sign_pair_idxs) > 0:
                sign_pair_idxs = torch.sort(sign_pair_idxs, dim=1)[0]
                _, unique_idx, inverse_idx = np.unique(sign_pair_idxs.data.cpu().numpy(), axis=0, return_index=True, return_inverse=True, return_counts=False)
                unique_idx = torch.from_numpy(unique_idx).to(pair_idx)
                inverse_idx = torch.from_numpy(inverse_idx).to(pair_idx)
            else:
                unique_idx = torch.empty((0,)).to(pair_idx)
                inverse_idx = torch.empty((0,)).to(pair_idx)
            unique_inverse_indices_list.append(inverse_idx)
            unique_indices_list.append(unique_idx)

            if len(sub_same_sign_pair_idx) > 0:
                assert obj_feat.shape[0] == len(proposal)
                assert pair_idx.shape[0] == rel_label_full.shape[0]
                assert int(sub_same_sign_pair_idx.max())+1 <= len(pair_idx)
                assert int(pair_idx.max())+1<=len(proposal)
                head_feat_sub_same = obj_feat[pair_idx[:, 0][sub_same_sign_pair_idx[:, 0]]]
                tail_feat_sub_same = union_feat[sub_same_sign_pair_idx[:, 1]]
                # TODO
                tail_bbox_sub_same = proposal[pair_idx[:, 0][sub_same_sign_pair_idx[:, 0]]]
                head_bbox_sub_same = boxlist_union(proposal[pair_idx[:, 0][sub_same_sign_pair_idx[:, 1]]],
                                        proposal[pair_idx[:, 1][sub_same_sign_pair_idx[:, 1]]])
                if self.training:
                    hmp_labels_sub_same = (rel_label_full[sub_same_sign_pair_idx[:, 0]] == rel_label_full[sub_same_sign_pair_idx[:, 1]]).to(obj_feat)
                else:
                    hmp_labels_sub_same = None
                # hmp_labels_sub_same = (hmp_labels_sub_same - 0.5) * 2.
            else:
                head_feat_sub_same = torch.empty((0, obj_feat.shape[-1])).to(obj_feat)
                tail_feat_sub_same = torch.empty((0, obj_feat.shape[-1])).to(obj_feat)
                tail_bbox_sub_same = None
                head_bbox_sub_same = None

            if len(obj_same_sign_pair_idx) > 0:
                assert obj_feat.shape[0] == len(proposal)
                assert pair_idx.shape[0] == rel_label_full.shape[0]
                assert int(obj_same_sign_pair_idx.max())+1 <= len(pair_idx)
                assert int(pair_idx.max())+1<=len(proposal)
                head_feat_obj_same = obj_feat[pair_idx[:, 0][obj_same_sign_pair_idx[:, 0]]]
                tail_feat_obj_same = union_feat[obj_same_sign_pair_idx[:, 1]]
                # TODO
                head_bbox_obj_same = proposal[pair_idx[:, 0][obj_same_sign_pair_idx[:, 0]]]
                tail_bbox_obj_same = boxlist_union(proposal[pair_idx[:, 0][obj_same_sign_pair_idx[:, 1]]],
                                        proposal[pair_idx[:, 1][obj_same_sign_pair_idx[:, 1]]])
                if self.training:
                    hmp_labels_obj_same = (rel_label_full[obj_same_sign_pair_idx[:, 0]] == rel_label_full[obj_same_sign_pair_idx[:, 1]]).to(obj_feat)
                else:
                    hmp_labels_obj_same = None
                # hmp_labels_obj_same = (hmp_labels_obj_same - 0.5) * 2.
            else:
                head_feat_obj_same = torch.empty((0, obj_feat.shape[-1])).to(obj_feat)
                tail_feat_obj_same = torch.empty((0, obj_feat.shape[-1])).to(obj_feat)
                tail_bbox_obj_same = None
                head_bbox_obj_same = None

            if tail_bbox_obj_same is None and tail_bbox_sub_same is None:
                head_bbox_list.append(None)
                tail_bbox_list.append(None)
                head_feat_list.append(torch.empty((0, obj_feat.shape[-1])).to(obj_feat))
                tail_feat_list.append(torch.empty((0, obj_feat.shape[-1])).to(obj_feat))
                hmp_labels_lists.append(torch.empty((0,)).to(obj_feat))
            elif tail_bbox_obj_same is None:
                head_bbox_list.append(head_bbox_sub_same[unique_idx])
                tail_bbox_list.append(tail_bbox_sub_same[unique_idx])
                head_feat_list.append(head_feat_sub_same)
                tail_feat_list.append(tail_feat_sub_same)
                hmp_labels_lists.append(hmp_labels_sub_same)
            elif tail_bbox_sub_same is None:
                head_bbox_list.append(head_bbox_obj_same[unique_idx])
                tail_bbox_list.append(tail_bbox_obj_same[unique_idx])
                head_feat_list.append(head_feat_obj_same)
                tail_feat_list.append(tail_feat_obj_same)
                hmp_labels_lists.append(hmp_labels_obj_same)
            else:
                head_bbox = cat_boxlist_nofield([head_bbox_sub_same, head_bbox_obj_same])[unique_idx]
                tail_bbox = cat_boxlist_nofield([tail_bbox_sub_same, tail_bbox_obj_same])[unique_idx]
                head_feat = cat([head_feat_sub_same, head_feat_obj_same], dim=0)
                tail_feat = cat([tail_feat_sub_same, tail_feat_obj_same], dim=0)
                if self.training:
                    hmp_labels = cat([hmp_labels_sub_same, hmp_labels_obj_same], dim=0)
                else:
                    hmp_labels = None
                head_bbox_list.append(head_bbox)
                tail_bbox_list.append(tail_bbox)
                head_feat_list.append(head_feat)
                tail_feat_list.append(tail_feat)
                hmp_labels_lists.append(hmp_labels)

        num_unique_pairs = [len(p) for p in unique_indices_list]
        union_feats = union_features_extractor.forward_bbox_pairs(features, head_bbox_list, tail_bbox_list)
        union_feats = union_feats.split(num_unique_pairs)
        union_feats = [x[idx] for x, idx in zip(union_feats, unique_inverse_indices_list)]
        union_feats = cat(union_feats, dim=0)
        num_pairs = [len(p) for p in hmp_pair_idxs_list]
        head_feats = cat(head_feat_list, dim=0)
        tail_feats = cat(tail_feat_list, dim=0)
        sign_factor = self.rel_hmp(self.hmp_ws(head_feats)*self.hmp_wo(tail_feats)*self.hmp_wu(union_feats)).squeeze(-1)
        if len(sign_factor) > 0 and self.training:
            labels = cat(hmp_labels_lists, 0)
            loss = F.binary_cross_entropy_with_logits(sign_factor, labels)
        else:
            loss = torch.zeros(1).to(obj_feats[0]).mean()

        sign_factor_list = self.tanh(sign_factor).split(num_pairs)

        return sign_factor_list, hmp_pair_idxs_list, loss

    def forward(self, proposals, rel_pair_idxs, pair_idxs, rel_labels,
                rel_binarys, roi_features, union_features, features, union_features_extractor, logger=None):

        obj_dists, obj_preds, obj_feats_nonlocal, atten_map_list, add_losses = self.context_layer(roi_features, proposals, union_features, pair_idxs, logger)

        obj_preds_embeds = self.ort_embedding.index_select(0, obj_preds.long()).to(roi_features)

        num_pairs = [p.shape[0] for p in pair_idxs]
        union_features = union_features.split(num_pairs, dim=0)
        if self.training:
            pair_idxs_new = []
            for pair_idx, rel_pair_idx in zip(pair_idxs, rel_pair_idxs):
                if len(rel_pair_idx) == 0:
                    pair_idxs_new.append(rel_pair_idx)
                else:
                    pair_idxs_new.append(pair_idx)
            pair_idxs = pair_idxs_new

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        num_pairs = [p.shape[0] for p in pair_idxs]

        assert len(num_rels) == len(num_objs)
        assert len(num_rels) == len(num_pairs)

        roi_features = roi_features.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_preds_embeds = obj_preds_embeds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        spt_feats = []
        obj_feats = []

        for rel_pair_idx, pair_idx, obj_pred, roi_feat, union_feat, obj_embed, bboxes, nonlocal_obj_feat in zip(
            rel_pair_idxs, pair_idxs, obj_preds, roi_features, union_features,
            obj_preds_embeds, proposals, obj_feats_nonlocal):
            if torch.numel(rel_pair_idx) == 0:
                if logger is not None:
                    logger.warning('image {} rel pair idx is emtpy!\nrel_pair_idx:{}\npair_idx:{}\nbboxes:{}'.format(
                        bboxes.image_fn, str(rel_pair_idx), str(pair_idx), str(bboxes)))
                prod_reps.append(torch.empty((0,self.pooling_dim)).to(roi_feat))
                pair_preds.append(torch.empty((0,2)).to(obj_pred))
                spt_feats.append(torch.empty((0,64)).to(roi_feat))
                obj_feats.append(torch.empty((0,self.pooling_dim)).to(roi_feat))
                continue
            w, h = bboxes.size
            bboxes_tensor = bboxes.bbox
            transfered_boxes = torch.stack(
                (
                    bboxes_tensor[:, 0] / w,
                    bboxes_tensor[:, 3] / h,
                    bboxes_tensor[:, 2] / w,
                    bboxes_tensor[:, 1] / h,
                    (bboxes_tensor[:, 2] - bboxes_tensor[:, 0]) * \
                    (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) / w / h,
                ), dim=-1
            ).to(roi_feat)
            obj_features_low = cat(
                (
                    roi_feat, obj_embed, transfered_boxes
                ), dim=-1
            )

            obj_features = self.merge_obj_low(obj_features_low) + self.merge_obj_high(nonlocal_obj_feat)

            obj_feats.append(obj_features)

            subj_rep, obj_rep = self.post_emb_s(obj_features), self.post_emb_o(obj_features)
            union_feat_rel = union_feat

            spt_feats.append( self.get_boxes_encode(bboxes_tensor, pair_idx, w, h) )
            prod_reps.append( subj_rep[pair_idx[:, 0]] * obj_rep[pair_idx[:, 1]] * union_feat_rel )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )

        prod_reps = cat(prod_reps, dim=0)
        pair_preds = cat(pair_preds, dim=0)
        spt_feats = cat(spt_feats, dim=0)

        # spt_feats = self.get_boxes_encode(proposals, rel_pair_idxs)

        prod_reps = cat((prod_reps, spt_feats), dim=-1)

        rel_dists = self.rel_compress(prod_reps)

        rel_dists_new = []
        pair_preds_new = []
        vr_indices_list = []
        rel_labels_full = []
        prod_reps_new = []

        if not self.training:
            rel_labels = [None] * len(num_pairs)

        for pair_idx, rel_pair_idx, rel_label in zip(pair_idxs, rel_pair_idxs, rel_labels):
            if len(pair_idx) == 0:
                rel_labels_full.append(torch.empty((0,)).to(pair_idx))
                vr_indices_list.append(torch.empty((0,)).to(pair_idx))
                continue
            if len(rel_pair_idx) > 0:
                vr_indices = self.intersect_2d_tensor(rel_pair_idx, pair_idx).argmax(-1)
            else:
                vr_indices = torch.empty((0,)).to(pair_idx)
            rel_label_full = torch.zeros_like(pair_idx[:, 0])
            if self.training:
                rel_label_full[vr_indices] = rel_label
            rel_labels_full.append(rel_label_full)
            vr_indices_list.append(vr_indices)

        sign_list, sign_pair_idxs_list, hmp_loss = self.get_hmp_infos(proposals, pair_idxs, rel_labels_full, obj_feats, union_features, features, union_features_extractor)

        if self.training:
            add_losses.update(dict(rel_hmp_loss=hmp_loss))

        for pair_idx, pair_pred, rel_dist, atten_map, proposal, sign, sign_pair_idxs, vr_indices, prod_rep in zip(
                pair_idxs, pair_preds.split(num_pairs), rel_dists.split(num_pairs), atten_map_list , proposals,
                sign_list, sign_pair_idxs_list, vr_indices_list, prod_reps.split(num_pairs)):
            if len(pair_idx) == 0:
                rel_dists_new.append(rel_dist)
                pair_preds_new.append(pair_pred)
                prod_reps_new.append(prod_rep)
                continue
            ious = boxlist_iou(proposal, proposal)
            ious = ious * (1-torch.eye(ious.shape[0])).to(ious)

            mask_sub_same = (pair_idx[:,0][:,None]==pair_idx[:,0][None,:]).to(atten_map)
            mask_obj_same = (pair_idx[:,1][:,None]==pair_idx[:,1][None,:]).to(atten_map)

            exist_same_sub = (mask_sub_same.sum(1) > 1).to(atten_map[0])[:,None]
            exist_same_obj = (mask_obj_same.sum(1) > 1).to(atten_map[0])[:,None]

            norm_factor = (exist_same_obj+exist_same_sub).clamp(min=1.)

            for i in range(5):
                if i == 0:
                    rel_dist_new = rel_dist
                atten_map_i = atten_map[i]
                atten_map_entropy=entropy(atten_map_i.data.cpu().numpy(), axis=1)
                atten_map_entropy=torch.from_numpy(atten_map_entropy).to(atten_map_i)
                entropy_pair_sum = atten_map_entropy[pair_idx[:, 0]][:, None] + atten_map_entropy[pair_idx[:, 1]][None, :]
                miu = 0.5
                beta = (miu - entropy_pair_sum / 2.).clamp(min=0.)
                beta = beta[pair_idx[:, 0], pair_idx[:, 1]]
                atten_obj_same = atten_map_i[pair_idx[:,0]][:,pair_idx[:,0]]
                atten_sub_same = atten_map_i[pair_idx[:,1]][:,pair_idx[:,1]]
                atten_map_i = mask_sub_same*atten_sub_same + mask_obj_same * atten_obj_same
                atten_map_i = atten_map_i / norm_factor


                rel_dist_new =  - torch.mm(torch.diag(beta), rel_dist_new + torch.mm(atten_map_i, rel_dist_new)) + torch.mm(torch.diag(1+beta), rel_dist)

            assert int(vr_indices.max())+1<=rel_dist_new.shape[0]
            assert int(vr_indices.max())+1<=pair_pred.shape[0]
            rel_dists_new.append(rel_dist_new[vr_indices])
            pair_preds_new.append(pair_pred[vr_indices])
            prod_reps_new.append(prod_rep[vr_indices])

        rel_dists = cat(rel_dists_new)
        pair_preds = cat(pair_preds_new)
        prod_reps = cat(prod_reps_new)

        if self.use_bias:
            freq_gate = torch.sigmoid(self.freq_gate(prod_reps))
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_preds.long()) * freq_gate

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)


        return obj_dists, rel_dists, add_losses



def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
    
    
    
    
    


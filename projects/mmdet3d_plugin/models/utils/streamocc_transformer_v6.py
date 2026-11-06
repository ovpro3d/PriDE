import torch
import torch.nn as nn
from mmdet.models.utils.builder import TRANSFORMER

from projects.mmdet3d_plugin.ops import MSMV_CUDA
from projects.mmdet3d_plugin.models.utils.misc import DUMP
from .streamocc_transformer_v4 import (
    StreamOccTransformerV4, SuperOccTransformerDecoderV4)
from .difficulty_scorer_v3 import PrimitiveScorerV3


class SuperOccTransformerDecoderV6(SuperOccTransformerDecoderV4):

    def forward(self, query_feat, query_points, temp_memory, temp_reference_points,
                memory_mask, mlvl_feats, data, img_metas):
        cls_scores_list, refine_sqs_list = [], []
        query_feat_list, s_scores_list, s_logits_list = [], [], []

        occ2img = data['ego2img'] if self.use_ego else data['lidar2img']

        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape
            N, T, G, C = self.num_views, self.num_frames, self.num_groups, GC // self.num_groups
            assert T * N == TN
            feat = feat.reshape(B, T, N, G, C, H, W)
            if MSMV_CUDA:
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)
                feat = feat.reshape(B * T * G, N, H, W, C)
            else:
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)
                feat = feat.reshape(B * T * G, C, N, H, W)
            mlvl_feats[lvl] = feat.contiguous()

        s_prev = None
        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i
            query_points = query_points.detach()

            query_feat, cls_score, refine_sqs = decoder_layer(
                query_feat, query_points, temp_memory, temp_reference_points,
                memory_mask, mlvl_feats, occ2img, img_metas, s_prev=s_prev)

            query_points = refine_sqs[..., :3]

            scorer = self.scorers[i] if self.per_layer_scorer else self.scorer
            cls_in = cls_score.detach() if getattr(scorer, 'use_cls', False) else None
            if getattr(self, 'detach_scorer_input', False):
                s, u = scorer(query_feat.detach(), refine_sqs.detach(), cls_logits=cls_in)
            else:
                s, u = scorer(query_feat, refine_sqs, cls_logits=cls_in)

            cls_scores_list.append(cls_score)
            refine_sqs_list.append(refine_sqs)
            query_feat_list.append(query_feat)
            s_scores_list.append(s)
            s_logits_list.append(u)

            if self.attn_detach_s:
                s_prev = self._rank01(0.5 * (s.detach().amax(dim=2) + s.detach().mean(dim=2)))
            else:
                s_prev = 0.5 * (s.amax(dim=2) + s.mean(dim=2))

            if i < self.num_layers - 1:
                if self.adapter_s_mode == 'off':
                    s_query = 0.5 * (s.detach().amax(dim=2) + s.detach().mean(dim=2))
                    query_feat = query_feat + 0.0 * self.adapters[i](query_feat, s_query)
                else:
                    if self.adapter_s_mode == 'random':
                        s_a = torch.rand_like(s)
                    elif self.adapter_s_mode == 'uniform':
                        s_a = torch.full_like(s, 0.5)
                    else:
                        s_a = s
                    if self.detach_s_in_adapter:
                        s_a = s_a.detach()
                    s_query = 0.5 * (s_a.amax(dim=2) + s_a.mean(dim=2))
                    if self.adapter_s_mode != 'uniform' and self.detach_s_in_adapter:
                        s_query = self._rank01(s_query)
                    query_feat = self.adapters[i](query_feat, s_query)

        if self.training:
            mod_tie = None
            for L in self.decoder_layers:
                sa, sp = getattr(L, 'self_attn', None), getattr(L, 'sampling', None)
                for p in (getattr(getattr(sa, 'mod', None), 'kappa', None),
                          getattr(getattr(sa, 'mod', None), 'beta', None),
                          getattr(getattr(sp, 'mod', None), 'kappa', None),
                          getattr(getattr(sp, 'mod', None), 'beta', None),
                          getattr(sp, 'level_bias', None)):
                    if p is not None:
                        mod_tie = p.sum() if mod_tie is None else mod_tie + p.sum()
            if mod_tie is not None:
                # Connect modulation parameters to both score outputs.
                s_scores_list[0] = s_scores_list[0] + mod_tie * 0.0
                s_logits_list[0] = s_logits_list[0] + mod_tie * 0.0

        return query_feat_list, cls_scores_list, refine_sqs_list, s_scores_list, s_logits_list


@TRANSFORMER.register_module()
class StreamOccTransformerV6(StreamOccTransformerV4):

    def __init__(self, *args, scorer_num_classes=0,
                 scorer_use_query_feat=True, scorer_use_params=True, 
                 detach_scorer_input=False,
                 **kwargs):
        super().__init__(*args, scorer_num_classes=scorer_num_classes, **kwargs)
        self.decoder.__class__ = SuperOccTransformerDecoderV6
        self.decoder.detach_scorer_input = detach_scorer_input
        embed_dims = self.decoder.decoder_layers[0].embed_dims
        if self.decoder.per_layer_scorer:
            self.decoder.scorers = nn.ModuleList([
                PrimitiveScorerV3(embed_dims, param_dim=13,
                                  num_classes=scorer_num_classes,
                                  use_query_feat=scorer_use_query_feat,
                                  use_params=scorer_use_params)
                for _ in range(self.decoder.num_layers)
            ])
        else:
            self.decoder.scorer = PrimitiveScorerV3(
                embed_dims, param_dim=13, num_classes=scorer_num_classes,
                use_query_feat=scorer_use_query_feat,
                use_params=scorer_use_params)

    def forward(self, query_feat, query_points, temp_memory, temp_reference_points,
                memory_mask, mlvl_feats, data, img_metas):
        query_feats, cls_scores, refine_pts, s_scores, s_logits = self.decoder(
            query_feat, query_points, temp_memory, temp_reference_points,
            memory_mask, mlvl_feats, data, img_metas)
        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]
        s_scores = [torch.nan_to_num(score) for score in s_scores]
        s_logits = [torch.nan_to_num(logit) for logit in s_logits]
        return query_feats, cls_scores, refine_pts, s_scores, s_logits

import torch
from mmdet.models.utils.builder import TRANSFORMER

from projects.mmdet3d_plugin.ops import MSMV_CUDA
from projects.mmdet3d_plugin.models.utils.misc import DUMP
from .streamocc_transformer_v6 import (
    StreamOccTransformerV6, SuperOccTransformerDecoderV6)
from .difficulty_modulation import ZeroSumModulator
from .difficulty_attn_v10 import SuperOccSelfAttentionV10, SuperOccSamplingV10


class SuperOccTransformerDecoderV10(SuperOccTransformerDecoderV6):

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
                    # zero-weight pass: keeps the adapter params in the DDP graph
                    if self.training:
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
                s_scores_list[0] = s_scores_list[0] + mod_tie * 0.0
                s_logits_list[0] = s_logits_list[0] + mod_tie * 0.0

        return query_feat_list, cls_scores_list, refine_sqs_list, s_scores_list, s_logits_list


@TRANSFORMER.register_module()
class StreamOccTransformerV10(StreamOccTransformerV6):

    def __init__(self, *args, attn_mode='learned', **kwargs):
        super().__init__(*args, attn_mode=attn_mode, **kwargs)
        self.decoder.__class__ = SuperOccTransformerDecoderV10
        num_levels = self.decoder.decoder_layers[0].sampling.num_levels
        self.zs_mod = ZeroSumModulator(num_levels, mode=attn_mode)
        for L in self.decoder.decoder_layers:
            sa, sp = L.self_attn, L.sampling
            L.self_attn = SuperOccSelfAttentionV10(
                L.embed_dims, num_heads=sa.num_heads, dropout=0.1,
                pc_range=L.pc_range, mod=self.zs_mod)
            L.sampling = SuperOccSamplingV10(
                L.embed_dims, num_frames=sp.num_frames, num_views=sp.num_views,
                num_groups=sp.num_groups, num_points=sp.num_points,
                num_levels=sp.num_levels, pc_range=L.pc_range, mod=self.zs_mod)

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList
from mmdet.models.utils.builder import TRANSFORMER

from projects.mmdet3d_plugin.ops import MSMV_CUDA
from projects.mmdet3d_plugin.models.utils.misc import DUMP
from .streamocc_transformer import (
    StreamOccTransformer, SuperOccTransformerDecoder, SuperOccTransformerDecoderLayer)
from .difficulty_scorer import PrimitiveScorer, DifficultyAdapter
from .difficulty_attn import SuperOccSelfAttentionV3, SuperOccSamplingV3
from .difficulty_modulation import rank01_tie_aware


class SuperOccTransformerDecoderLayerV4(SuperOccTransformerDecoderLayer):
    def __init__(self, *args, attn_mode='learned', detach_s=True, attn_init_gain=-4.0, **kwargs):
        super().__init__(*args, **kwargs)
        sa, sp = self.self_attn, self.sampling
        self.self_attn = SuperOccSelfAttentionV3(
            self.embed_dims, num_heads=sa.num_heads, dropout=0.1,
            pc_range=self.pc_range, attn_mode=attn_mode, detach_s=detach_s,
            init_gain=attn_init_gain)
        self.sampling = SuperOccSamplingV3(
            self.embed_dims, num_frames=sp.num_frames, num_views=sp.num_views,
            num_groups=sp.num_groups, num_points=sp.num_points,
            num_levels=sp.num_levels, pc_range=self.pc_range,
            attn_mode=attn_mode, detach_s=detach_s, init_gain=attn_init_gain)

    def forward(self, query_feat, query_points, temp_memory, temp_reference_points, memory_mask,
                mlvl_feats, occ2img, img_metas, s_prev=None):
        query_pos = self.position_encoder(query_points.flatten(2, 3))
        sampled_feat = self.sampling(
            query_feat, query_pos, query_points, mlvl_feats, occ2img, img_metas, s=s_prev)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat, query_pos))
        query_feat = self.norm2(self.self_attn(query_feat, query_pos, query_points, s=s_prev))
        query_feat = self.norm3(self.ffn(query_feat))

        B, Q = query_points.shape[:2]
        cls_score = self.cls_branch(query_feat)
        reg_offset = self.reg_branch(query_feat).reshape(B, Q, self.num_refines, -1)
        reg_offset = self.scale(reg_offset)
        cls_score = cls_score.unsqueeze(dim=2).repeat(1, 1, self.num_refines, 1)
        refine_sqs = self.refine_sqs(query_points, reg_offset)
        return query_feat, cls_score, refine_sqs


class SuperOccTransformerDecoderV4(SuperOccTransformerDecoder):
    def __init__(self, *args, scorer_num_classes=0,
                 detach_s_in_adapter=True, adapter_s_mode='learned',
                 adapter_condition_s=True, adapter_warmup_iters=500,
                 attn_mode='learned', attn_detach_s=True, attn_init_gain=-4.0,
                 per_layer_scorer=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert adapter_s_mode in ('learned', 'random', 'uniform', 'off')
        self.detach_s_in_adapter = detach_s_in_adapter
        self.adapter_s_mode = adapter_s_mode
        self.attn_mode = attn_mode
        self.attn_detach_s = attn_detach_s
        self.attn_init_gain = attn_init_gain
        self.per_layer_scorer = per_layer_scorer
        embed_dims = self.decoder_layers[0].embed_dims
        if per_layer_scorer:
            self.scorers = nn.ModuleList([
                PrimitiveScorer(embed_dims, param_dim=13, num_classes=scorer_num_classes)
                for _ in range(self.num_layers)
            ])
        else:
            self.scorer = PrimitiveScorer(embed_dims, param_dim=13, num_classes=scorer_num_classes)

        self.adapters = nn.ModuleList([
            DifficultyAdapter(embed_dims,
                              condition_on_s=adapter_condition_s,
                              warmup_iters=adapter_warmup_iters)
            for _ in range(max(0, self.num_layers - 1))
        ])
        new_layers = ModuleList()
        for L in self.decoder_layers:
            sp = L.sampling
            new_layers.append(SuperOccTransformerDecoderLayerV4(
                L.embed_dims, sp.num_frames, sp.num_views, L.num_points, sp.num_levels,
                L.num_classes, sp.num_groups, L.num_refines, L.last_refines,
                pc_range=L.pc_range, attn_mode=attn_mode, detach_s=attn_detach_s,
                attn_init_gain=attn_init_gain))
        self.decoder_layers = new_layers

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()

    @staticmethod
    def _rank01(x):
        return rank01_tie_aware(x)

    def forward(self, query_feat, query_points, temp_memory, temp_reference_points, memory_mask,
                mlvl_feats, data, img_metas):
        cls_scores_list, refine_sqs_list = [], []
        query_feat_list, s_scores_list = [], []

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
            s = scorer(query_feat, refine_sqs)               # (B, Q, N_refine) in (0,1)

            cls_scores_list.append(cls_score)
            refine_sqs_list.append(refine_sqs)
            query_feat_list.append(query_feat)               # store feature that produced THIS layer's outputs
            s_scores_list.append(s)

            if self.attn_detach_s:
                s_prev = self._rank01(0.5 * (s.detach().amax(dim=2) + s.detach().mean(dim=2)))
            else:
                s_prev = 0.5 * (s.amax(dim=2) + s.mean(dim=2))

            if i < self.num_layers - 1:
                if self.adapter_s_mode == 'off':
                    
                    s_query = 0.5 * (s.detach().amax(dim=2) + s.detach().mean(dim=2))
                    query_feat = query_feat + 0.0 * self.adapters[i](query_feat, s_query)
                else:
                    if self.adapter_s_mode == 'random':      # ablation: difficulty -> noise
                        s_a = torch.rand_like(s)
                    elif self.adapter_s_mode == 'uniform':   # ablation: plain residual block
                        s_a = torch.full_like(s, 0.5)
                    else:
                        s_a = s
                    if self.detach_s_in_adapter:
                        s_a = s_a.detach()
                    s_query = 0.5 * (s_a.amax(dim=2) + s_a.mean(dim=2))   # (B, Q)
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

        return query_feat_list, cls_scores_list, refine_sqs_list, s_scores_list


@TRANSFORMER.register_module()
class StreamOccTransformerV4(StreamOccTransformer):

    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=[1, 2, 4, 8, 16, 32],
                 use_ego=True,
                 pc_range=[],
                 num_pt_channels=None,
                 scorer_num_classes=0,
                 detach_s_in_adapter=True,
                 adapter_s_mode='learned',
                 adapter_condition_s=True,
                 adapter_warmup_iters=500,
                 attn_mode='learned',
                 attn_detach_s=True,
                 attn_init_gain=-4.0,
                 per_layer_scorer=False,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_points = num_points
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.use_ego = use_ego
        self.num_pt_channels = num_pt_channels if num_pt_channels is not None else num_classes
        self.decoder = SuperOccTransformerDecoderV4(
            embed_dims, num_frames, num_views, num_points, num_layers, num_levels,
            num_classes, num_refines, num_groups, use_ego, pc_range=pc_range,
            scorer_num_classes=scorer_num_classes,
            detach_s_in_adapter=detach_s_in_adapter,
            adapter_s_mode=adapter_s_mode,
            adapter_condition_s=adapter_condition_s,
            adapter_warmup_iters=adapter_warmup_iters,
            attn_mode=attn_mode,
            attn_detach_s=attn_detach_s,
            attn_init_gain=attn_init_gain,
            per_layer_scorer=per_layer_scorer)

    def forward(self, query_feat, query_points, temp_memory, temp_reference_points, memory_mask,
                mlvl_feats, data, img_metas):
        query_feats, cls_scores, refine_pts, s_scores = self.decoder(
            query_feat, query_points, temp_memory, temp_reference_points, memory_mask,
            mlvl_feats, data, img_metas)
        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]
        return query_feats, cls_scores, refine_pts, s_scores

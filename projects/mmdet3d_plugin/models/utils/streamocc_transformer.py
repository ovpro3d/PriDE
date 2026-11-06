import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from mmcv.cnn import bias_init_with_prob, Scale
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN
from mmdet.models.utils.builder import TRANSFORMER
from projects.mmdet3d_plugin.core.bbox.util import decode_points, encode_points
from projects.mmdet3d_plugin.models.utils.misc import safe_sigmoid
from .superocc_sampling import sampling_4d
from .checkpoint import checkpoint as cp
from projects.mmdet3d_plugin.ops import MSMV_CUDA
from projects.mmdet3d_plugin.models.utils.misc import DUMP


@TRANSFORMER.register_module()
class StreamOccTransformer(BaseModule):
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
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.num_refines = num_refines

        self.decoder = SuperOccTransformerDecoder(
            embed_dims, num_frames, num_views, num_points, num_layers, num_levels,
            num_classes, num_refines, num_groups, use_ego, pc_range=pc_range)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_feat, query_points, temp_memory, temp_reference_points, memory_mask,
                mlvl_feats, data, img_metas):
        """
        Args:
            query_feat: (B, N_query, C)
            query_points: (B, N_query, 1, 3)
            temp_memory: (B, Mem, C)
            temp_reference_points: (B, Mem, 3)
            mlvl_feats: List[(B, N, C=256, H2, W2), ..., (B, N, C=256, H5, W5)]
        Returns:
            cls_scores: List[(B, N_query, n_refine_1, n_cls), (B, N_query, n_refine_2, n_cls), ...]
            refine_pts: List[(B, N_query, n_refine_1, 3), (B, N_query, n_refine_2, 3), ...]
        """
        query_feats, cls_scores, refine_pts = self.decoder(
            query_feat, query_points, temp_memory, temp_reference_points, memory_mask, mlvl_feats, data, img_metas)

        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]

        return query_feats, cls_scores, refine_pts


class SuperOccTransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=1,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_refines=16,
                 num_groups=4,
                 use_ego=True,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_groups = num_groups
        self.use_ego = use_ego

        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
        last_refines = [1] + num_refines

        # params are shared across all decoder layers
        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                SuperOccTransformerDecoderLayer(
                    embed_dims, num_frames, num_views, num_points, num_levels, num_classes,
                    num_groups, num_refines[i], last_refines[i], pc_range=pc_range)
            )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()

    def forward(self, query_feat, query_points, temp_memory, temp_reference_points, memory_mask,
                mlvl_feats, data, img_metas):
        """
        Args:
            query_feat: (B, N_query, C)
            query_points: (B, N_query, n_refine, 3)
            temp_memory: (B, Mem, C)
            temp_reference_points: (B, Mem, 3)
            mlvl_feats: List[(B, N, C=256, H2, W2), ..., (B, N, C=256, H5, W5)]
        Returns:
            cls_scores: List[(B, N_query, n_refine_1, n_cls), (B, N_query, n_refine_2, n_cls), ...]
            refine_pts: List[(B, N_query, n_refine_1, 3), (B, N_query, n_refine_2, 3), ...]
        """
        cls_scores_list, refine_sqs_list = [], []
        query_feat_list = []

        # organize projections matrix and copy to CUDA
        occ2img = data['ego2img'] if self.use_ego else data['lidar2img']

        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # (B, T*N_view, GC, fH, fW)
            N, T, G, C = self.num_views, self.num_frames, self.num_groups, GC//self.num_groups
            assert T*N == TN
            # (B, N, C=256, H2, W2) --> (B, T, N_view, G, C, fH, fW)
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                # (B, T, N_view, G, C, fH, fW) --> (B, T, G, N_view, fH, fW, C)
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)
                # (B*T*G, N_view, fH, fW, C)
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                # (B, T, N_view, G, C, fH, fW) --> (B, T, G, C, N_view, fH, fW)
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)
                # (B*T*G, C, N_view, fH, fW)
                feat = feat.reshape(B*T*G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()
        # mlvl_feats: List[(B*T*G, N_view, H2, W2, C), (B*T*G, N_view, H2, W2, C), ...]

        for i, decoder_layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i

            query_points = query_points.detach()
            # query_feat: (B, N_query, N_refine, C)
            # cls_score:  (B, N_query, N_refine, n_cls)
            # refine_sqs: (B, N_query, N_refine, 13)
            query_feat, cls_score, refine_sqs = decoder_layer(
                query_feat,  # (B, N_query, C)
                query_points,   # (B, N_query, N_refine, 3)
                temp_memory,    # (B, Mem, C)
                temp_reference_points,   # (B, Mem, 3)
                memory_mask,
                mlvl_feats,     # List[(B*T*G, N_view, H2, W2, C), (B*T*G, N_view, H2, W2, C), ...]
                occ2img,        # (B, N=T*N_view, 4, 4)
                img_metas
            )

            query_points = refine_sqs[..., :3]  # (B, N_query, N_refine, 3)

            cls_scores_list.append(cls_score)
            refine_sqs_list.append(refine_sqs)
            query_feat_list.append(query_feat)

        return query_feat_list, cls_scores_list, refine_sqs_list


class SuperOccTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=1,
                 num_views=6,
                 num_points=4,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=16,
                 last_refines=16,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.num_points = num_points
        self.num_refines = num_refines
        self.last_refines = last_refines

        self.position_encoder = nn.Sequential(
            nn.Linear(3*self.last_refines, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = SuperOccSelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = SuperOccSampling(embed_dims, num_frames=num_frames, num_views=num_views,
                                         num_groups=num_groups, num_points=num_points,
                                         num_levels=num_levels, pc_range=pc_range)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames,
                                     n_groups=num_groups, out_points=num_points * num_frames)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(
            self.embed_dims, self.num_classes))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, 13 * self.num_refines))
        self.reg_branch = nn.Sequential(*reg_branch)

        self.output_dim = 13
        self.scale = Scale([1.0] * self.output_dim)
        self.register_buffer("unit_xyz", torch.tensor([3.0, 3.0, 3.0], dtype=torch.float))

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_sqs(self, points_proposal, points_delta):
        """
        Args:
            points_proposal: (B, N_query, N_refine, 3)
            points_delta: (B, N_query, N_refine', 13)
        Returns:
            refine_sqs: (B, N_query, N_refine‘, 13)
        """
        B, Q = points_delta.shape[:2]
        # (B, N_query, N_refine‘ * 3) --> (B, N_query, N_refine’, 13)
        points_delta = points_delta.reshape(B, Q, self.num_refines, 13)

        points_proposal = decode_points(points_proposal, self.pc_range)     # (B, N_query, N_refine, 3)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)     # (B, N_query, 1, 3)
        # new_points = points_proposal + points_delta[..., :3]    # (B, N_query, N_refine, 3)
        delta_xyz = (2 * safe_sigmoid(points_delta[..., :3]) - 1.) * self.unit_xyz[None, None, None]
        new_points = points_proposal + delta_xyz  # (B, N_query, N_refine, 3)

        xyz = encode_points(new_points, self.pc_range)  # (B, N_query, N_refine‘, 3)
        xyz = torch.clamp(xyz, min=1e-6, max=1-1e-6)

        scale = points_delta[..., 3:6]  # (B, N_query， N_refine’, 3)
        rot = torch.nn.functional.normalize(points_delta[..., 6:10], p=2, dim=-1)  # (B, N_query，N_refine‘, 4)
        feat = points_delta[..., 10:]  # (B, N_query， N_refine’, 3)
        refine_sqs = torch.cat([xyz, scale, rot, feat], dim=-1)     # (B, N_query， N_refine‘, 13)
        return refine_sqs

    def forward(self, query_feat, query_points, temp_memory, temp_reference_points, memory_mask,
                mlvl_feats, occ2img, img_metas):
        """
        Args:
            query_feat: (B, N_query, C)
            query_points: (B, N_query, N_refine, 3)
            temp_memory: (B, Mem, C)
            temp_reference_points: (B, Mem, 3)
            mlvl_feats: List[(B*T*G, N_view, H2, W2, C), (B*T*G, N_view, H2, W2, C), ...]
            occ2img: (B, N=T*N_view, 4, 4)
        Returns:
            query_feat: (B, N_query, C)
            cls_score:  (B, N_query, N_refine, n_cls)
            refine_pt:  (B, N_query, N_refine, 3)
        """
        # (B, N_query, N_refine*3) --> (B, N_query, C)
        query_pos = self.position_encoder(query_points.flatten(2, 3))

        sampled_feat = self.sampling(
            query_feat,  # (B, N_query, C)
            query_pos,   # (B, N_query, C)
            query_points,   # (B, N_query, N_refine, 3)
            mlvl_feats,     # List[(B*T*G, N_view, H2, W2, C), (B*T*G, N_view, H2, W2, C), ...]
            occ2img,        # (B, N=T*N_view, 4, 4)
            img_metas
        )   # (B, N_query, n_group, T*n_points, C)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat, query_pos))      # (B, N_query, C)
        query_feat = self.norm2(self.self_attn(query_feat, query_pos, query_points))  # (B, N_query, C)
        query_feat = self.norm3(self.ffn(query_feat))   # (B, N_query, C)

        B, Q = query_points.shape[:2]
        cls_score = self.cls_branch(query_feat)   # (B, N_query, n_refine * n_cls)
        reg_offset = self.reg_branch(query_feat)  # (B, N_query, n_refine * 13)
        reg_offset = reg_offset.reshape(B, Q, self.num_refines, -1)     # (B, N_query, n_refine, 13)
        reg_offset = self.scale(reg_offset)

        # (B, N_query, n_refine * n_cls) --> (B, N_query, n_refine, n_cls)
        # cls_score = cls_score.reshape(B, Q, self.num_refines, self.num_classes)
        cls_score = cls_score.unsqueeze(dim=2).repeat(1, 1, self.num_refines, 1)
        refine_sqs = self.refine_sqs(query_points, reg_offset)    # (B, N_query, N_refine, 13)

        if DUMP.enabled:
            pass # TODO: enable OTR dump

        return query_feat, cls_score, refine_sqs


class SuperOccSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.1,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range
        self.num_heads = num_heads

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_feat, query_pos, reference_points):
        """
        Args:
            query_feat: (B, Q, C)
            query_pos:  (B, Q, C)
            reference_points: (B, Q, n_refine, 3)

        Return:
            query_feat: (B, Q, C)
        """
        reference_points = reference_points.mean(dim=2)
        temp_key = temp_value = query_feat
        temp_reference_points = reference_points
        temp_pos = query_pos

        dist = self.calc_points_dists(reference_points, temp_reference_points)     # (B, Q, K)
        tau = self.gen_tau(query_feat)  # (B, Q, n_head)

        # if DUMP.enabled:
        #     torch.save(tau.cpu(), '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)      # (B, n_head, Q)
        attn_mask = dist[:, None, :, :] * tau[..., None]  # (B, n_head, Q, K)
        attn_mask = attn_mask.flatten(0, 1)  # (Bxn_head, Q, K)

        return self.attention(query_feat, temp_key, temp_value, identity=None, query_pos=query_pos,
                              key_pos=temp_pos, attn_mask=attn_mask, key_padding_mask=None)

    def forward(self, query_feat, query_pos, query_points):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_feat, query_pos, query_points, use_reentrant=False)
        else:
            return self.inner_forward(query_feat, query_pos, query_points)

    @torch.no_grad()
    def calc_points_dists(self, points, temp_points):
        """
        Args:
            points: (B, Q, 3)
            temp_points: (B, K, 3)
        Returns:
            -dist: (B, Q, K)
        """
        points = decode_points(points, self.pc_range)   # (B, Q, 3)
        temp_points = decode_points(temp_points, self.pc_range)   # (B, K, 3)
        # (B, Q, 1, 3) - (B, 1, K, 3)  --> (B, Q, K)
        dist = torch.norm(points.unsqueeze(-2) - temp_points.unsqueeze(-3), dim=-1)
        return -dist


class SuperOccSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self,
                 embed_dims=256,
                 num_frames=4,
                 num_views=6,
                 num_groups=4,
                 num_points=8,
                 num_levels=4,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_views = num_views
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_feat, query_pos, query_points, mlvl_feats, occ2img, img_metas):
        '''
        Args:
            query_feat: (B, N_query, C)
            query_pos: (B, N_query, C)
            query_points: (B, N_query, N_refine, 3)
            mlvl_feats: List[(B*T*G, N_view, H2, W2, C), (B*T*G, N_view, H2, W2, C), ...]
            occ2img: (B, N=T*N_view, 4, 4)
        Returns:
            sampled_feats: (B, N_query, n_group, T*n_points, C)
        '''
        query_feat = query_feat + query_pos
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        # query points
        query_points = decode_points(query_points, self.pc_range)   # (B, N_query, N_refine, 3)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.zeros_like(query_center)
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        # sampling offset of all frames
        # (B, N_query, C=256) --> (B, N_query, n_group*n_points*3=48)
        sampling_offset = self.sampling_offset(query_feat)
        # (B, N_query, N_group*N_points*3=48) --> (B, N_query, N_group*N_points=4x4=16, 3)
        sampling_offset = sampling_offset.view(B, Q, -1, 3)

        # sampling_points = query_center + sampling_offset * query_scale  # (B, N_query, N_group*N_points, 3)
        sampling_points = query_center + sampling_offset  # (B, N_query, N_group*N_points, 3)
        # (B, N_query, N_group*N_points, 3) --> (B, N_query, N_group=4, N_points=4, 3)
        sampling_points = sampling_points.view(B, Q, self.num_groups, self.num_points, 3)
        # (B, N_query, N_group=4, N_points=4, 3) --> (B, N_query, 1, N_group=4, N_points=4, 3)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        # (B, N_query, 1, N_group=4, N_points=4, 3) --> (B, N_query, T, N_group=4, N_points=4, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # scale weights
        # (B, N_query, C=256) --> (B, N_query, n_group*n_points*n_levels)
        # --> (B, N_query, n_group, 1, n_points, n_levels)
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        # (B, N_query, n_group, T, n_points, n_levels)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,    # (B, N_query, T, N_group=4, N_points=4, 3)
            mlvl_feats,         # List[(B*T*G, N_view, H2, W2, C), (B*T*G, N_view, H2, W2, C), ...]
            scale_weights,      # (B, N_query, n_group=4, T, n_points, n_levels)
            occ2img,            # (B, N=T*N_view, 4, 4)
            image_h, image_w,
            self.num_views
        )  # (B, N_query, n_group, T*n_points, C)

        return sampled_feats

    def forward(self, query_feat, query_pos, query_points, mlvl_feats, occ2img, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_feat, query_pos, query_points, mlvl_feats,
                      occ2img, img_metas, use_reentrant=False)
        else:
            return self.inner_forward(query_feat, query_pos, query_points, mlvl_feats,
                                      occ2img, img_metas)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super().__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query, query_pos):
        """
        Args:
            x: (B, Q, G, P=N_frames*N_points, C)
            query: (B, Q, C)
            query_pos: (B, Q, C)
        Returns:
            out: (B, Q, C)
        """
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        # (B, N_query, C)  --> (B, Q, G*(64*64+32*32))
        params = self.parameter_generator(query + query_pos)
        params = params.reshape(B*Q, G, -1)     # (B, Q, G, 64*64+32*32=5120)
        out = x.reshape(B*Q, G, P, C)   # (B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)    # (B*Q, G, C_in=64, C_out=64)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)      # (B*Q, G, out_points=32, in_points=32)

        '''adaptive channel mixing'''
        # (B*Q, G, P, C_in) @ (B*Q, G, C_in, C_out) --> (B*Q, G, P, C_out)
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        # (B*Q, G, out_P, in_P) @ (B*Q, G, in_P, C_out) --> (B*Q, G, out_P, C_out)
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)     # (B*Q, G, out_P, C_out) --> (B, Q, G*out_P*C_out)
        out = self.out_proj(out)        # (B, Q, G*out_P*C_out) --> (B, Q, C)
        out = query + out

        return out

    def forward(self, x, query, query_pos):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, query_pos, use_reentrant=False)
        else:
            return self.inner_forward(x, query, query_pos)
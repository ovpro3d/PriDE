import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.mmdet3d_plugin.core.bbox.util import decode_points
from .superocc_sampling import sampling_4d
from .checkpoint import checkpoint as cp
from .streamocc_transformer import SuperOccSelfAttention, SuperOccSampling


class DiffAttnMod(nn.Module):
    def __init__(self, attn_mode='learned', detach_s=True, init_gain=-4.0):
        super().__init__()
        assert attn_mode in ('learned', 'random', 'uniform', 'off')
        self.attn_mode = attn_mode
        self.detach_s = detach_s
        self.kappa = nn.Parameter(torch.full((1,), float(init_gain)))  # query temperature gain
        self.beta = nn.Parameter(torch.full((1,), float(init_gain)))   # key bias gain

    def _prep(self, s):
        if s is None or self.attn_mode == 'off':
            return None
        if self.attn_mode == 'random':
            s = torch.rand_like(s)
        elif self.attn_mode == 'uniform':
            s = torch.full_like(s, 0.5)
        return s.detach() if self.detach_s else s

    def temp(self, s):
        s = self._prep(s)
        if s is None:
            return None
        return 1.0 + F.softplus(self.kappa) * s

    def keybias(self, s):
        s = self._prep(s)
        if s is None:
            return None
        return F.softplus(self.beta) * s


class SuperOccSelfAttentionV3(SuperOccSelfAttention):
    def __init__(self, *args, attn_mode='learned', detach_s=True, init_gain=-4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mod = DiffAttnMod(attn_mode, detach_s, init_gain=init_gain)

    def inner_forward(self, query_feat, query_pos, reference_points, s=None):
        reference_points = reference_points.mean(dim=2)
        temp_key = temp_value = query_feat
        temp_pos = query_pos

        dist = self.calc_points_dists(reference_points, reference_points)   # (B, Q, K)
        tau = self.gen_tau(query_feat).permute(0, 2, 1)                     # (B, n_head, Q)

        if s is not None:
            T = self.mod.temp(s)                       # (B, Q) or None
            if T is not None:
                tau = tau / T[:, None, :].clamp_min(1e-4)

        attn_mask = dist[:, None, :, :] * tau[..., None]                    # (B, n_head, Q, K)

        if s is not None:
            kb = self.mod.keybias(s)                   # (B, K) or None
            if kb is not None:
                attn_mask = attn_mask + kb[:, None, None, :]

        attn_mask = attn_mask.flatten(0, 1)                                 # (B*n_head, Q, K)
        return self.attention(query_feat, temp_key, temp_value, identity=None,
                              query_pos=query_pos, key_pos=temp_pos,
                              attn_mask=attn_mask, key_padding_mask=None)

    def forward(self, query_feat, query_pos, query_points, s=None):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_feat, query_pos, query_points, s, use_reentrant=False)
        return self.inner_forward(query_feat, query_pos, query_points, s)


class SuperOccSamplingV3(SuperOccSampling):
    def __init__(self, *args, attn_mode='learned', detach_s=True, init_gain=-4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mod = DiffAttnMod(attn_mode, detach_s, init_gain=init_gain)
        self.level_bias = nn.Parameter(torch.zeros(self.num_levels))

    def inner_forward(self, query_feat, query_pos, query_points, mlvl_feats, occ2img, img_metas, s=None):
        query_feat = query_feat + query_pos
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
        else:
            query_center = query_points.mean(dim=2, keepdim=True)

        sampling_offset = self.sampling_offset(query_feat).view(B, Q, -1, 3)
        if s is not None:
            T = self.mod.temp(s)
            if T is not None:
                sampling_offset = sampling_offset * T[..., None, None]      # widen reach for hard
        sampling_points = query_center + sampling_offset
        sampling_points = sampling_points.view(B, Q, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        sw = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        if s is not None:
            T = self.mod.temp(s)
            if T is not None:
                sw = sw / T[:, :, None, None, None, None].clamp_min(1e-4)   # softer level mix
            kb = self.mod.keybias(s)
            if kb is not None:
                sw = sw + kb[:, :, None, None, None, None] * self.level_bias  # bias toward (learned) hi-res
        scale_weights = torch.softmax(sw, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        return sampling_4d(sampling_points, mlvl_feats, scale_weights, occ2img,
                           image_h, image_w, self.num_views)

    def forward(self, query_feat, query_pos, query_points, mlvl_feats, occ2img, img_metas, s=None):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_feat, query_pos, query_points,
                      mlvl_feats, occ2img, img_metas, s, use_reentrant=False)
        return self.inner_forward(query_feat, query_pos, query_points, mlvl_feats, occ2img, img_metas, s)

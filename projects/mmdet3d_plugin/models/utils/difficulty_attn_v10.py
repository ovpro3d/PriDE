import torch
import torch.nn.functional as F

from projects.mmdet3d_plugin.core.bbox.util import decode_points
from .superocc_sampling import sampling_4d
from .difficulty_attn import SuperOccSelfAttentionV3, SuperOccSamplingV3
from .difficulty_modulation import ZeroSumModulator


class SuperOccSelfAttentionV10(SuperOccSelfAttentionV3):
    def __init__(self, *args, mod=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(mod, ZeroSumModulator), 'pass the shared ZeroSumModulator'
        del self.mod                          # drop V3's per-module DiffAttnMod
        object.__setattr__(self, 'mod', mod)  # Reference the shared module without registering it again.

    def inner_forward(self, query_feat, query_pos, reference_points, s=None):
        reference_points = reference_points.mean(dim=2)
        temp_key = temp_value = query_feat
        temp_pos = query_pos
        dist = self.calc_points_dists(reference_points, reference_points)   # (B, Q, K)
        tau = self.gen_tau(query_feat).permute(0, 2, 1)                     # (B, n_head, Q)

        if s is not None and self.mod.mode != 'off':
            rho = self.mod.rho(s)                                           # (B, Q)
            T = self.mod.temperature(rho)                                   # (B, Q)
            tau = tau / T[:, None, :]                                       # Eq.11 first term
            attn_mask = dist[:, None, :, :] * tau[..., None]                # (B, n_head, Q, K)
            attn_mask = attn_mask + F.softplus(self.mod.beta) \
                * rho[:, None, :, None] * (1.0 - rho[:, None, None, :])
        else:
            attn_mask = dist[:, None, :, :] * tau[..., None]
        attn_mask = attn_mask + self.mod._ddp_tie()

        attn_mask = attn_mask.flatten(0, 1)                                 # (B*n_head, Q, K)
        return self.attention(query_feat, temp_key, temp_value, identity=None,
                              query_pos=query_pos, key_pos=temp_pos,
                              attn_mask=attn_mask, key_padding_mask=None)


class SuperOccSamplingV10(SuperOccSamplingV3):

    def __init__(self, *args, mod=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(mod, ZeroSumModulator), 'pass the shared ZeroSumModulator'
        del self.mod                          # drop V3's per-module DiffAttnMod
        object.__setattr__(self, 'mod', mod)  # Reference the shared module without registering it again.
        del self.level_bias   # b_lvl lives in the shared modulator (Eq.13)

    def inner_forward(self, query_feat, query_pos, query_points, mlvl_feats,
                      occ2img, img_metas, s=None):
        query_feat = query_feat + query_pos
        B, Q = query_points.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
        else:
            query_center = query_points.mean(dim=2, keepdim=True)

        rho = None
        if s is not None and self.mod.mode != 'off':
            rho = self.mod.rho(s)                                           # (B, Q)

        sampling_offset = self.sampling_offset(query_feat).view(B, Q, -1, 3)
        if rho is not None:
            sampling_offset = self.mod.modulate_offsets(sampling_offset, rho)  # Eq.12
        sampling_points = query_center + sampling_offset
        sampling_points = sampling_points.view(B, Q, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups,
                                                 self.num_points, 3)

        sw = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1,
                                                 self.num_points, self.num_levels)
        if rho is not None:
            sw = self.mod.modulate_level_logits(sw, rho)                    # Eq.13
        sw = sw + self.mod._ddp_tie()
        scale_weights = torch.softmax(sw, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames,
                                             self.num_points, self.num_levels)

        return sampling_4d(sampling_points, mlvl_feats, scale_weights, occ2img,
                           image_h, image_w, self.num_views)

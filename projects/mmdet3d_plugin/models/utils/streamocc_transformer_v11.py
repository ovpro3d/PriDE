import torch
import torch.nn.functional as F
from torch import nn
from mmdet.models.utils.builder import TRANSFORMER

from .streamocc_transformer_v10 import StreamOccTransformerV10
from .difficulty_attn_v10 import SuperOccSelfAttentionV10
from .difficulty_modulation import ZeroSumModulator


class _NoOpAdapter(nn.Module):
    """Parameter-free stand-in for a disabled DifficultyAdapter.

    The zero `gate` is a buffer, not a parameter: loss() reads
    `tanh(adapter.gate)` for its diag_gate entry, which thus stays 0.0."""

    def __init__(self):
        super().__init__()
        self.register_buffer('gate', torch.zeros(1))

    def forward(self, query_feat, s_query=None):
        return query_feat.new_zeros(())


class SuperOccSelfAttentionV11(SuperOccSelfAttentionV10):

    def __init__(self, *args, attn_topk=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_topk = int(attn_topk)

    def inner_forward(self, query_feat, query_pos, reference_points, s=None):
        reference_points = reference_points.mean(dim=2)
        temp_key = temp_value = query_feat
        temp_pos = query_pos

        dist = self.calc_points_dists(reference_points, reference_points)   # (B, Q, K)
        tau = self.gen_tau(query_feat).permute(0, 2, 1)                     # (B, n_head, Q)

        if s is not None and self.mod.mode != 'off':
            rho = self.mod.rho(s)                                           # (B, Q)
            T = self.mod.temperature(rho)
            tau = tau / T[:, None, :]              # Eq. (11): temperature modulation for all queries.
            attn_mask = dist[:, None, :, :] * tau[..., None]
            B, Q = rho.shape
            k = self.attn_topk
            if k == 0:
                pass          # key-side term disabled: temperature-only modulation
            elif 0 < k < Q:
                idx = rho.topk(k, dim=1).indices                            # (B, k)
                beta_sp = F.softplus(self.mod.beta)
                key_row = 1.0 - rho[:, None, None, :]                       # (B,1,1,K)
                for b in range(B):
                    attn_mask[b, :, idx[b], :] += \
                        beta_sp * rho[b, idx[b]][None, :, None] * key_row[b]
            else:             # k < 0 or k >= Q: full query-key bias
                attn_mask = attn_mask + F.softplus(self.mod.beta) \
                    * rho[:, None, :, None] * (1.0 - rho[:, None, None, :])
        else:
            attn_mask = dist[:, None, :, :] * tau[..., None]

        # Keep shared modulation parameters connected to the graph.
        attn_mask = attn_mask + self.mod._ddp_tie()

        attn_mask = attn_mask.flatten(0, 1)
        return self.attention(query_feat, temp_key, temp_value, identity=None,
                              query_pos=query_pos, key_pos=temp_pos,
                              attn_mask=attn_mask, key_padding_mask=None)


@TRANSFORMER.register_module()
class StreamOccTransformerV11(StreamOccTransformerV10):

    def __init__(self, *args, attn_mode='learned', attn_topk=-1, **kwargs):
        super().__init__(*args, attn_mode=attn_mode, **kwargs)
        layers = self.decoder.decoder_layers
        ks = list(attn_topk) if isinstance(attn_topk, (list, tuple)) \
            else [attn_topk] * len(layers)
        assert len(ks) == len(layers), \
            f'attn_topk list length {len(ks)} != num decoder layers {len(layers)}'
        for layer, k in zip(layers, ks):
            sa = layer.self_attn
            layer.self_attn = SuperOccSelfAttentionV11(
                layer.embed_dims, num_heads=sa.num_heads, dropout=0.1,
                pc_range=layer.pc_range, mod=self.zs_mod, attn_topk=int(k))


        dec = self.decoder
        if getattr(dec, 'adapter_s_mode', None) == 'off' \
                and getattr(dec, 'adapters', None) is not None:
            dec.adapters = nn.ModuleList([_NoOpAdapter() for _ in dec.adapters])

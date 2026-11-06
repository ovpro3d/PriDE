import torch
import torch.nn as nn
import torch.nn.functional as F


def rank01_tie_aware(x, valid=None, dim=-1):
    x = x.detach()
    assert dim == -1, 'rank01_tie_aware expects the population on the last dim'
    out = torch.full_like(x, 0.5)
    flat_x = x.reshape(-1, x.shape[-1])
    flat_v = valid.reshape(-1, x.shape[-1]) if valid is not None else None
    flat_o = out.view(-1, x.shape[-1])
    for b in range(flat_x.shape[0]):
        m = flat_v[b] > 0 if flat_v is not None else torch.ones_like(flat_x[b], dtype=torch.bool)
        n = int(m.sum())
        if n < 2:
            continue
        xb = flat_x[b][m]
        xs, _ = torch.sort(xb)
        lo = torch.searchsorted(xs, xb, right=False).float()
        hi = torch.searchsorted(xs, xb, right=True).float()
        flat_o[b][m] = (0.5 * (lo + hi) + 0.5) / float(n)
    return out


class ZeroSumModulator(nn.Module):

    MODES = ('learned', 'uniform', 'random', 'off')

    def __init__(self, num_levels, mode='learned',
                 init_kappa=-1.0, init_beta=-2.0, init_beta_s=-2.0):
        super().__init__()
        assert mode in self.MODES
        self.mode = mode
        self.kappa = nn.Parameter(torch.tensor(float(init_kappa)))
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        self.beta_s = nn.Parameter(torch.tensor(float(init_beta_s)))
        self.b_lvl = nn.Parameter(torch.zeros(num_levels))

    # -- score -> percentile ------------------------------------------------
    def rho(self, s, valid=None):
        if self.mode == 'uniform':
            return torch.full_like(s.detach(), 0.5)
        if self.mode == 'random':
            return rank01_tie_aware(torch.rand_like(s.detach()), valid)
        return rank01_tie_aware(s, valid)          # 'learned' and 'off'

    def _ddp_tie(self):
        return 0.0 * (self.kappa + self.beta + self.beta_s + self.b_lvl.sum())

    def temperature(self, rho):
        if self.mode == 'off':
            return torch.ones_like(rho) + self._ddp_tie()
        centered = rho - rho.mean(dim=-1, keepdim=True)
        return torch.exp(F.softplus(self.kappa) * centered)

    # -- Eq.(11): self-attention bias ---------------------------------------
    def attn_bias(self, dist, tau, rho):
        """dist: (B, Q, Q); tau: (B, Q) or (B, Q, 1); rho: (B, Q)."""
        tau = tau if tau.dim() == dist.dim() - 1 else tau.squeeze(-1)
        base = -dist * tau.unsqueeze(-1)
        if self.mode == 'off':
            return base + self._ddp_tie()
        T = self.temperature(rho)                               # (B, Q)
        bias = -dist * (tau / T).unsqueeze(-1)
        key_term = F.softplus(self.beta) * rho.unsqueeze(-1) * (1.0 - rho.unsqueeze(-2))
        return bias + key_term

    # -- Eq.(12): sampling offsets -------------------------------------------
    def modulate_offsets(self, offsets, rho):
        if self.mode == 'off':
            return offsets + self._ddp_tie()
        T = self.temperature(rho)
        return offsets * T.view(*T.shape, *([1] * (offsets.dim() - T.dim())))

    # -- Eq.(13): level logits -----------------------------------------------
    def modulate_level_logits(self, level_logits, rho):
        """level_logits: (..., num_levels) with rho matching its leading dims."""
        if self.mode == 'off':
            return level_logits + self._ddp_tie()
        T = self.temperature(rho)
        centered = rho - rho.mean(dim=-1, keepdim=True)
        expand = [1] * (level_logits.dim() - T.dim() - 1)
        Tx = T.view(*T.shape, *expand, 1)
        cx = centered.view(*centered.shape, *expand, 1)
        return level_logits / Tx + F.softplus(self.beta_s) * cx * self.b_lvl

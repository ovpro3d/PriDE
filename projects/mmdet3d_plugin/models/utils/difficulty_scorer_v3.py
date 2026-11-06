import torch
import torch.nn as nn
import torch.nn.functional as F

from .difficulty_scorer_v2 import (
    DifficultyAdapter, per_voxel_difficulty, voxel_structural_difficulty,
    rank_normalize, scorer_ranking_loss, spearman_corr)


class PrimitiveScorerV3(nn.Module):
    def __init__(self, embed_dims, param_dim=13, num_classes=0, hidden=128,
                 use_query_feat=True, use_params=True):
        super().__init__()
        assert use_query_feat or use_params or num_classes > 0, \
            'PrimitiveScorerV3 needs at least one input source.'
        self.use_query_feat = use_query_feat
        self.use_params = use_params
        self.use_cls = num_classes > 0
        in_dim = (embed_dims if use_query_feat else 0) \
            + (param_dim if use_params else 0) \
            + (num_classes if self.use_cls else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, query_feat, refine_sqs, cls_logits=None):
        B, Q, R, _ = refine_sqs.shape
        feats = []
        if self.use_query_feat:
            qf = query_feat.unsqueeze(2).expand(B, Q, R, query_feat.size(-1))
            feats.append(qf)
        if self.use_params:
            feats.append(refine_sqs)
        if self.use_cls:
            if cls_logits is None:
                raise ValueError('cls_logits is required when num_classes > 0.')
            feats.append(cls_logits)
        x = torch.cat(feats, dim=-1)
        logit = self.net(x).squeeze(-1)
        return torch.sigmoid(logit), logit


def _rank_percentile_ascending(x):
    xs, _ = torch.sort(x)
    lo = torch.searchsorted(xs, x, right=False).float()
    hi = torch.searchsorted(xs, x, right=True).float()
    return (0.5 * (lo + hi) + 0.5) / x.numel()


def paper_logistic_ranking_loss(logits, e_target, valid=None, gamma=1.0,
                                pair_mode='sampled', num_pairs=4096,
                                exact_chunk=1024):
    assert pair_mode in ('sampled', 'exact')
    B, P = logits.shape
    dev = logits.device
    valid_mask = valid > 0 if valid is not None else torch.ones(
        (B, P), dtype=torch.bool, device=dev)

    loss_num = logits.new_zeros(())
    weight_den = logits.new_zeros(())
    pairs_per_batch = max(1, int(num_pairs) // max(1, B))

    for b in range(B):
        m = valid_mask[b]
        if m.sum() < 2:
            continue
        u = logits[b, m].float()
        e = e_target[b, m].detach().float()
        rank = _rank_percentile_ascending(e)
        n = u.numel()

        if pair_mode == 'sampled':
            a = torch.randint(0, n, (pairs_per_batch,), device=dev)
            c = torch.randint(0, n, (pairs_per_batch,), device=dev)
            keep = a != c
            if keep.sum() == 0:
                continue
            a, c = a[keep], c[keep]
            ra, rc = rank[a], rank[c]
            keep = ra != rc
            if keep.sum() == 0:
                continue
            a, c, ra, rc = a[keep], c[keep], ra[keep], rc[keep]
            hard = torch.where(ra > rc, a, c)
            easy = torch.where(ra > rc, c, a)
            w = (ra - rc).abs()
            loss_num = loss_num + (w * F.softplus(-gamma * (u[hard] - u[easy]))).sum()
            weight_den = weight_den + w.sum()
        else:
            for start in range(0, n, exact_chunk):
                stop = min(start + exact_chunk, n)
                u_i = u[start:stop, None]
                r_i = rank[start:stop, None]
                r_j = rank[None, :]
                mask = r_i > r_j
                if not mask.any():
                    continue
                w = (r_i - r_j).masked_select(mask)
                diff = (u_i - u[None, :]).masked_select(mask)
                loss_num = loss_num + (w * F.softplus(-gamma * diff)).sum()
                weight_den = weight_den + w.sum()

    if weight_den <= 0:
        return logits.sum() * 0.0
    return loss_num / weight_den.clamp_min(1e-6)


__all__ = [
    'DifficultyAdapter', 'PrimitiveScorerV3',
    'per_voxel_difficulty', 'voxel_structural_difficulty',
    'rank_normalize', 'scorer_ranking_loss', 'spearman_corr',
    'paper_logistic_ranking_loss',
]

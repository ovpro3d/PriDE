
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimitiveScorer(nn.Module):

    def __init__(self, embed_dims, param_dim=13, num_classes=0, hidden=128):
        super().__init__()
        in_dim = embed_dims + param_dim + num_classes
        self.use_cls = num_classes > 0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, query_feat, refine_sqs, cls_logits=None):
        B, Q, R, _ = refine_sqs.shape
        qf = query_feat.unsqueeze(2).expand(B, Q, R, query_feat.size(-1))
        feats = [qf, refine_sqs]
        if self.use_cls and cls_logits is not None:
            feats.append(cls_logits)
        x = torch.cat(feats, dim=-1)
        s = torch.sigmoid(self.net(x)).squeeze(-1)   # (B, Q, R)
        return s


class DifficultyAdapter(nn.Module):

    def __init__(self, embed_dims, hidden=None, condition_on_s=True, warmup_iters=500):
        super().__init__()
        hidden = hidden or embed_dims
        self.condition_on_s = condition_on_s
        in_dim = embed_dims + (1 if condition_on_s else 0)
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dims),
        )
        self.gate = nn.Parameter(torch.zeros(1))   # tanh(gate) starts at 0
        self.warmup_iters = max(1, int(warmup_iters))
        self.register_buffer('_step', torch.zeros(1))

    def forward(self, query_feat, s_query):
        if self.training:
            self._step += 1
        warm = (self._step / self.warmup_iters).clamp(max=1.0)
        x = query_feat
        if self.condition_on_s:
            x = torch.cat([query_feat, s_query.unsqueeze(-1)], dim=-1)
        delta = self.adapter(x)
        g = torch.tanh(self.gate) * warm
        return query_feat + g * s_query.unsqueeze(-1) * delta

@torch.no_grad()
def voxel_structural_difficulty(gt, ignore_label):
    valid = (gt != ignore_label)
    chg = torch.zeros(gt.shape, dtype=torch.float32, device=gt.device)
    cnt = torch.zeros_like(chg)
    for dim in (-3, -2, -1):
        sl_a = [slice(None)] * gt.dim()
        sl_b = [slice(None)] * gt.dim()
        sl_a[dim] = slice(None, -1)
        sl_b[dim] = slice(1, None)
        a, b = tuple(sl_a), tuple(sl_b)
        pair = valid[a] & valid[b]
        diff = ((gt[a] != gt[b]) & pair).float()
        pairf = pair.float()
        chg[a] += diff
        chg[b] += diff
        cnt[a] += pairf
        cnt[b] += pairf
    return chg / cnt.clamp_min(1.0)


def per_voxel_difficulty(occ_prob, gt, ignore_label,
                         lovasz_fn=None, lam_lovasz=0.0,
                         camera_mask=None, h_loc=None, lam_struct=0.0):
    N, C = occ_prob.shape
    valid = (gt != ignore_label)
    if camera_mask is not None:
        valid = valid & (camera_mask.reshape(-1) > 0)
    gtc = gt.clamp(0, C - 1)

    ce = occ_prob.new_zeros(N)
    p_true = occ_prob.gather(1, gtc[:, None]).squeeze(1).clamp_min(1e-6)
    ce = torch.where(valid, -torch.log(p_true), ce)

    d = ce
    if lam_lovasz > 0.0 and lovasz_fn is not None:
        logits = occ_prob.detach().clone().requires_grad_(True)
        L = lovasz_fn(logits, gt)                       # scalar for this sample
        grad, = torch.autograd.grad(L, logits)          # (N, C)
        d = d + lam_lovasz * grad.abs().sum(dim=1)

    if h_loc is not None and lam_struct > 0.0:
        d = d * (1.0 + lam_struct * h_loc.reshape(-1).float())

    return (d * valid.float()).detach(), valid.float()


def rank_normalize(e, valid=None, eps=1e-6):
    B, P = e.shape
    out = torch.zeros_like(e)
    for b in range(B):
        x = e[b]
        if valid is not None:
            m = valid[b] > 0
            if m.sum() < 2:
                continue
            r = torch.argsort(torch.argsort(x[m]))
            out[b, m] = (r.float() + 1.0) / m.sum().float()
        else:
            r = torch.argsort(torch.argsort(x))
            out[b] = (r.float() + 1.0) / P
    return out.detach()


def scorer_ranking_loss(s_pred, e_target, valid=None, num_pairs=4096, margin=0.1):
    B, P = s_pred.shape
    s = s_pred.reshape(-1)
    t = e_target.reshape(-1).detach()
    dev = s.device
    i = torch.randint(0, B * P, (num_pairs,), device=dev)
    j = torch.randint(0, B * P, (num_pairs,), device=dev)
    if valid is not None:
        vv = valid.reshape(-1)
        keep = (vv[i] > 0) & (vv[j] > 0)
        i, j = i[keep], j[keep]
    if i.numel() == 0:
        return s.sum() * 0.0
    sign = torch.sign(t[i] - t[j])                 # +1 if i is harder
    loss = F.relu(margin - sign * (s[i] - s[j]))
    return loss.mean()


@torch.no_grad()
def spearman_corr(s, e, valid=None):
    rs = rank_normalize(s, valid)
    rt = rank_normalize(e, valid)
    m = (valid > 0) if valid is not None else torch.ones_like(s, dtype=torch.bool)
    if m.sum() < 2:
        return s.new_zeros(())
    a = rs[m] - rs[m].mean()
    b = rt[m] - rt[m].mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-6)

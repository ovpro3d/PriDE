# GPU parity test for tile_local_aggregate_prob_sq_bt2 (requires CUDA):
#
#   cd projects/mmdet3d_plugin/ops/tile_localagg_prob_sq_bt2
#   python test_backtrace_own_parity.py
#
# Checks, against a dense fp64 PyTorch reference that replicates the kernel math
# (rect test, superquadric f with the f>30 cutoff, power, prob, r_n = prob/prob_sum
# with the FORWARD-cached prob_sum as denominator):
#   [1] shared-CE kernel (local_aggregate_backtrace)     num/den  ==  reference
#   [2] own-CE  kernel (local_aggregate_backtrace_own)   num/den  ==  reference
#   [3] den(shared) == den(own) on the same valid mask (teacher support unchanged)
#   [4] outputs are finite, detached (requires_grad == False), and non-negative den
#   [5] forward-output parity between bt2 and bt, when bt is installed.
#
# Expected tolerance: atomicAdd reorders fp32 sums -> rel. diff O(1e-5); the test
# uses rtol=1e-4, atol=1e-4 on num and den.

import math
import torch

from tile_local_aggregate_prob_sq_bt2 import (
    LocalAggregatorExtV2, local_aggregate_backtrace, local_aggregate_backtrace_own)

torch.manual_seed(0)
DEV = 'cuda'

# Square x/y grid required by the CUDA voxel indexing (H == W).
H = W = 40
D = 16
GRID = 0.4
PC_MIN = [0.0, 0.0, 0.0]
P = 96                  # primitives
C_TOT = 18              # 17 occupied classes + appended zero free column
N_CLS = C_TOT - 1
IGNORE, FREE = 255, 17
EPS1, EPS2 = 1e-6, 1e-6


def build_scene():
    xs = torch.arange(H, device=DEV)
    ys = torch.arange(W, device=DEV)
    zs = torch.arange(D, device=DEV)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing='ij')
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).float() * GRID + GRID / 2.0
    pts += torch.tensor(PC_MIN, device=DEV)

    extent = torch.tensor([H * GRID, W * GRID, D * GRID], device=DEV)
    means = torch.tensor(PC_MIN, device=DEV) + 0.05 * extent \
        + torch.rand(P, 3, device=DEV) * 0.9 * extent
    scales = 0.3 + torch.rand(P, 3, device=DEV) * 1.2
    q = torch.randn(P, 4, device=DEV)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    rot = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        dim=-1).reshape(P, 3, 3)
    opas = 0.05 + 0.95 * torch.rand(P, device=DEV)
    u = 0.5 + 1.5 * torch.rand(P, device=DEV)
    v = 0.5 + 1.5 * torch.rand(P, device=DEV)
    sem = torch.softmax(torch.randn(P, N_CLS, device=DEV) * 2.0, dim=-1)
    sem = torch.cat([sem, torch.zeros(P, 1, device=DEV)], dim=-1)   # (P, 18)

    gt = torch.randint(0, N_CLS, (H * W * D,), device=DEV)
    gt[torch.rand(H * W * D, device=DEV) < 0.55] = FREE
    gt[torch.rand(H * W * D, device=DEV) < 0.05] = IGNORE
    return pts, means, scales, rot, opas, u, v, sem, gt


@torch.no_grad()
def dense_reference(cache, sem, gt, valid, use_own, ce_shared=None):
    """fp64 dense replication of the kernel math (same rect test, f cutoff)."""
    pts = cache['pts'].double()                      # (N, 3)
    means = cache['means3D'].double()                # (P, 3)
    mint = cache['means3D_int']                      # (P, 3) int
    scales = cache['scales3D'].double()
    rot = cache['rot3D'].double().reshape(-1, 3, 3)  # (P, 3, 3) row-major rows
    opas = cache['opas'].double()
    uu = cache['u'].double()
    vv = cache['v'].double()
    radii = cache['radii']
    prob_sum = cache['probability'].double()         # (N,) forward-cached denominator

    Np = pts.size(0)
    vox = torch.stack([torch.div(torch.arange(Np, device=DEV), H * D, rounding_mode='floor'),
                       torch.div(torch.arange(Np, device=DEV) % (H * D), D, rounding_mode='floor'),
                       torch.arange(Np, device=DEV) % D], dim=-1)   # kernel's (x, y, z)

    num = torch.zeros(P, dtype=torch.float64, device=DEV)
    den = torch.zeros(P, dtype=torch.float64, device=DEV)
    act = (prob_sum > 1e-9) & (valid > 0.5)

    for n in range(P):
        r = int(radii[n])
        lo = torch.tensor([max(0, int(mint[n, 0]) - r), max(0, int(mint[n, 1]) - r),
                           max(0, int(mint[n, 2]) - r)], device=DEV)
        hi = torch.tensor([min(W, int(mint[n, 0]) + r + 1), min(H, int(mint[n, 1]) + r + 1),
                           min(D, int(mint[n, 2]) + r + 1)], device=DEV)
        inside = ((vox >= lo) & (vox < hi)).all(dim=-1) & act
        if not inside.any():
            continue
        d = pts[inside] - means[n]
        t = d @ rot[n].T                                  # rows of rot3D are the kernel's rot1/2/3
        tx = ((t[:, 0] / scales[n, 0]) ** 2) ** (1.0 / uu[n])
        ty = ((t[:, 1] / scales[n, 1]) ** 2) ** (1.0 / uu[n])
        tz = ((t[:, 2] / scales[n, 2]) ** 2) ** (1.0 / vv[n])
        f = (tx + ty) ** (uu[n] / vv[n]) + tz
        keep = torch.isfinite(f) & (f <= 30.0)
        if not keep.any():
            continue
        power = torch.exp(-0.5 * f[keep])
        resp = power * opas[n] / prob_sum[inside][keep]
        if use_own:
            y = gt[inside][keep].clamp(0, C_TOT - 1)
            d_own = -torch.log(sem[n].double()[y] + EPS1)
            num[n] = (resp * d_own).sum()
        else:
            num[n] = (resp * ce_shared[inside][keep].double()).sum()
        den[n] = resp.sum()
    return num, den


def main():
    pts, means, scales, rot, opas, u, v, sem, gt = build_scene()

    agg = LocalAggregatorExtV2(scale_multiplier=3, H=H, W=W, D=D,
                               pc_min=PC_MIN, grid_size=GRID).to(DEV)
    logits, bin_logits, density = agg(
        pts[None], means[None], opas[None], u[None], v[None],
        sem[None], scales[None], rot.reshape(1, P, 3, 3))
    cache = agg._bt_cache
    print(f'forward ok: R={cache["R"]}, prob_sum>0 on '
          f'{(cache["probability"] > 1e-9).float().mean():.1%} of voxels')

    # ---- shared-CE teacher (V7/V9 path) ----
    aggregated = logits[:, :N_CLS] * bin_logits.unsqueeze(-1)
    aggregated = torch.cat([aggregated, (1 - bin_logits).unsqueeze(-1)], dim=-1)
    valid_shared = (gt != IGNORE).float()
    p_true = aggregated.gather(1, gt.clamp(0, C_TOT - 1)[:, None]).squeeze(1).clamp_min(1e-6)
    ce_shared = (-torch.log(p_true)) * valid_shared

    num_s, den_s = local_aggregate_backtrace(cache, ce_shared, valid_shared)
    ref_num_s, ref_den_s = dense_reference(cache, sem, gt, valid_shared,
                                           use_own=False, ce_shared=ce_shared)

    # ---- own-CE teacher (V10 paper path); paper valid = occupied & non-ignored ----
    valid_own = ((gt != IGNORE) & (gt != FREE)).float()
    num_o, den_o = local_aggregate_backtrace_own(cache, gt, valid_own, eps1=EPS1)
    ref_num_o, ref_den_o = dense_reference(cache, sem, gt, valid_own, use_own=True)
    # den(shared) reference on the SAME (occupied-only) mask for check [3]
    num_s2, den_s2 = local_aggregate_backtrace(cache, ce_shared * valid_own, valid_own)

    def report(name, a, b):
        a32, b32 = a.float(), b.float()
        diff = (a32 - b32).abs()
        rel = (diff / b32.abs().clamp_min(1e-8)).max()
        print(f'  {name:14s} max|diff|={diff.max():.3e}  max rel={rel:.3e}')
        assert torch.allclose(a32, b32, rtol=1e-4, atol=1e-4), f'{name} mismatch'

    print('[1] shared-CE kernel vs dense fp64 reference')
    report('num', num_s, ref_num_s); report('den', den_s, ref_den_s)
    print('[2] own-CE kernel vs dense fp64 reference')
    report('num', num_o, ref_num_o); report('den', den_o, ref_den_o)
    print('[3] den parity across teachers on the same valid mask')
    report('den', den_o, den_s2)
    print('[4] detached & finite')
    assert not num_o.requires_grad and not den_o.requires_grad
    assert torch.isfinite(num_o).all() and torch.isfinite(den_o).all() and (den_o >= 0).all()
    e = num_o / (den_o + EPS2)
    print(f'  e_own: min={e.min():.4f} max={e.max():.4f} '
          f'(bound -log(eps1)={-math.log(EPS1):.4f})')
    assert e.max() <= -math.log(EPS1) + 1e-3

    try:
        from tile_local_aggregate_prob_sq_bt import LocalAggregatorExt
        agg_old = LocalAggregatorExt(scale_multiplier=3, H=H, W=W, D=D,
                                     pc_min=PC_MIN, grid_size=GRID).to(DEV)
        lo_, bo_, do_ = agg_old(pts[None], means[None], opas[None], u[None], v[None],
                                sem[None], scales[None], rot.reshape(1, P, 3, 3))
        print('[5] bt2 forward vs bt forward')
        report('logits', logits, lo_); report('bin_logits', bin_logits, bo_)
    except ImportError:
        print('[5] old bt package not installed -- skipped (kernels are byte-identical)')

    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()

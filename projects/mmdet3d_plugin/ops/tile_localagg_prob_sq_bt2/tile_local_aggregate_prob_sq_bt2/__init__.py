# Superquadric aggregation with shared-CE and per-primitive CE backtrace.
# The own-CE teacher accumulates A_n = sum_v r_n(v) * w(v) * (-log(p_{n,y_v} + eps1))
# and D_n = sum_v r_n(v) over valid voxels; e_n = A_n / (D_n + eps2).

import torch
import torch.nn as nn
from . import _C


class _LocalAggregateExt(torch.autograd.Function):
    """Aggregate occupancy and expose non-differentiable buffers for backtrace."""
    @staticmethod
    def forward(ctx, pts, points_int, means3D, means3D_int, opas, u, v,
                semantics, scales3D, rot3D, radii, H, W, D):
        args = (pts, points_int, means3D, means3D_int, opas, u, v,
                semantics, scales3D, rot3D, radii, H, W, D)
        num_rendered, logits, bin_logits, density, probability, \
            geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args)

        ctx.num_rendered = num_rendered
        ctx.H = H; ctx.W = W; ctx.D = D
        ctx.save_for_backward(
            geomBuffer, binningBuffer, imgBuffer, means3D, means3D_int, pts,
            points_int, scales3D, rot3D, opas, u, v, radii, semantics,
            logits, bin_logits, density, probability)
        ctx.mark_non_differentiable(probability, geomBuffer, binningBuffer, imgBuffer)
        return logits, bin_logits, density, probability, geomBuffer, binningBuffer, imgBuffer, num_rendered

    @staticmethod
    def backward(ctx, logits_grad, bin_logits_grad, density_grad, *rest):
        geomBuffer, binningBuffer, imgBuffer, means3D, means3D_int, pts, \
            points_int, scales3D, rot3D, opas, u, v, radii, semantics, \
            logits, bin_logits, density, probability = ctx.saved_tensors
        args = (geomBuffer, binningBuffer, imgBuffer, ctx.H, ctx.W, ctx.D,
                ctx.num_rendered, means3D, means3D_int, pts, points_int,
                scales3D, rot3D, opas, u, v, radii, semantics,
                logits, bin_logits, density, probability,
                logits_grad, bin_logits_grad, density_grad)
        means3D_grad, opas_grad, u_grad, v_grad, semantics_grad, \
            rot3D_grad, scales3D_grad = _C.local_aggregate_backward(*args)
        return (None, None, means3D_grad, None, opas_grad, u_grad, v_grad,
                semantics_grad, scales3D_grad, rot3D_grad, None, None, None, None)


class LocalAggregatorExtV2(nn.Module):
    """Return (logits, bin_logits, density) and cache inputs for both backtrace modes.

    The own-CE teacher indexes cached input semantics by the voxel ground-truth label.
    """

    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size
        self.radii_min = radii_min

    def forward(self, pts, means3D, opas, u, v, semantics, scales, rot3D):
        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.squeeze(0)
        opas = opas.squeeze(0)
        u = u.squeeze(0)
        v = v.squeeze(0)
        semantics = semantics.squeeze(0)
        scales3D = scales.clone().squeeze(0)
        scales = scales.detach().squeeze(0)
        rot3D = rot3D.squeeze(0)

        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int)
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int)
        # Check grid bounds before conversion to unsigned CUDA indices.
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H \
            and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H \
            and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        radii = radii.clamp(min=self.radii_min)
        assert radii.min() >= 1
        rot3D = rot3D.flatten(1)

        logits, bin_logits, density, probability, geomBuffer, binningBuffer, imgBuffer, R = \
            _LocalAggregateExt.apply(
                pts, points_int, means3D, means3D_int, opas, u, v,
                semantics, scales3D, rot3D, radii, self.H, self.W, self.D)

        # Cache detached inputs, class probabilities, and buffers for backtrace.
        self._bt_cache = dict(
            binningBuffer=binningBuffer, imgBuffer=imgBuffer, R=int(R),
            probability=probability.detach(),
            pts=pts.detach(), points_int=points_int,
            means3D=means3D.detach(), means3D_int=means3D_int,
            scales3D=scales3D.detach(), rot3D=rot3D.detach(),
            opas=opas.detach(), u=u.detach(), v=v.detach(), radii=radii,
            semantics=semantics.detach(),
            H=self.H, W=self.W, D=self.D)
        return logits, bin_logits, density


@torch.no_grad()
def local_aggregate_backtrace(cache, ce, valid):
    """Aggregate a per-voxel scalar by primitive responsibility; return (num, den).

    The kernels bound vox.x by W and index voxels as x*H*D + y*D + z.
    This indexing requires H == W (for example, a 200x200x16 grid).
    """
    num, den = _C.local_aggregate_backtrace(
        cache['binningBuffer'], cache['imgBuffer'],
        cache['H'], cache['W'], cache['D'], cache['R'],
        cache['means3D'], cache['means3D_int'], cache['pts'],
        cache['scales3D'], cache['rot3D'], cache['opas'], cache['u'], cache['v'], cache['radii'],
        cache['probability'], ce.contiguous().float(), valid.contiguous().float())
    return num, den


@torch.no_grad()
def local_aggregate_backtrace_own(cache, gt, valid, wvox=None, eps1=1e-6):
    """PriDE paper teacher (Backtraced Error Teacher), fully in CUDA.

    For every cached (primitive n, voxel v) candidate pair the kernel recomputes the
    forward kernel/contribution and accumulates, over valid voxels:
        num[n] += r_n(v) * w(v) * ( -log(p_{n, y_v} + eps1) )     # A_n
        den[n] += r_n(v)                                          # D_n
    where p_{n, y_v} is primitive n's class probability at the voxel's GT label
    (indexed from the cached forward-input `semantics`).

    Args:
      cache: LocalAggregatorExtV2._bt_cache of the same forward call.
      gt:    (N,) long/int voxel GT labels in the aggregator's voxel order.
             Callers must pass labels < C_tot; valid voxels should be occupied
             (paper: V_valid = non-ignored occupied voxels), since the appended
             free column of `semantics` is identically 0.
      valid: (N,) float/bool mask in {0,1}.
      wvox:  optional (N,) per-voxel weight (default all-ones = plain paper teacher;
             allows CE * (1 + lam_struct * h_loc) style variants).
      eps1:  the paper's numerical-stability constant inside the log.

    Returns (num, den); difficulty e_n = num / (den + eps2) computed by the caller.
    Both outputs are detached (no autograd path -- the teacher must not send
    gradients to the primitives)."""
    sem = cache['semantics']
    C_tot = sem.size(1)
    gt_i = gt.reshape(-1).clamp(0, C_tot - 1).to(torch.int32).contiguous()
    if wvox is None:
        wvox = torch.ones_like(cache['probability'])
    num, den = _C.local_aggregate_backtrace_own(
        cache['binningBuffer'], cache['imgBuffer'],
        cache['H'], cache['W'], cache['D'], cache['R'],
        cache['means3D'], cache['means3D_int'], cache['pts'],
        cache['scales3D'], cache['rot3D'], cache['opas'], cache['u'], cache['v'], cache['radii'],
        sem.contiguous().float(), cache['probability'],
        gt_i, valid.contiguous().float(), wvox.contiguous().float(), float(eps1))
    return num, den


__all__ = ['LocalAggregatorExtV2', 'local_aggregate_backtrace', 'local_aggregate_backtrace_own']

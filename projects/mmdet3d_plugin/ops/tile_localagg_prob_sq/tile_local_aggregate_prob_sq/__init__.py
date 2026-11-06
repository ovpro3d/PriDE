#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
import torch.nn.functional as F

_C_STANDARD = None
_C_DGA = None
_STANDARD_IMPORT_ERROR = None
_DGA_IMPORT_ERROR = None

try:
    from . import _C as _C_STANDARD
except ImportError as err:
    _STANDARD_IMPORT_ERROR = err

try:
    from . import _C_dga as _C_DGA
except ImportError as err:
    _DGA_IMPORT_ERROR = err


def _normalize_backend(backend):
    if backend is None:
        return 'standard'

    backend_name = str(backend).lower()
    if backend_name in ('standard', 'default', 'base', '_c'):
        return 'standard'
    if backend_name in ('dga', '_c_dga'):
        return 'dga'

    raise ValueError(
        f"Unsupported aggregator backend '{backend}'. "
        "Expected one of: standard, dga."
    )


def _get_backend_module(backend):
    backend_name = _normalize_backend(backend)
    if backend_name == 'standard' and _C_STANDARD is not None:
        return _C_STANDARD, backend_name
    if backend_name == 'dga' and _C_DGA is not None:
        return _C_DGA, backend_name

    available = []
    if _C_STANDARD is not None:
        available.append('standard')
    if _C_DGA is not None:
        available.append('dga')

    if backend_name == 'standard':
        backend_error = _STANDARD_IMPORT_ERROR
    else:
        backend_error = _DGA_IMPORT_ERROR

    raise RuntimeError(
        f"Requested aggregator backend '{backend_name}' is not available. "
        f"Available backends: {available if available else ['none']}. "
        "Rebuild the extension in projects/ via `python setup.py develop`. "
        f"Original import error: {backend_error}"
    )


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opas,
        u, v,
        semantics,
        scales3D,
        rot3D,
        radii,
        H,
        W,
        D,
        backend
    ):

        backend_module, backend_name = _get_backend_module(backend)

        # Restructure arguments the way that the C++ lib expects them
        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            opas,
            u, v,
            semantics,
            scales3D,
            rot3D,
            radii,
            H, W, D
        )
        # Invoke C++/CUDA rasterizer
        num_rendered, logits, bin_logits, density, probability, geomBuffer, binningBuffer, imgBuffer = backend_module.local_aggregate(*args) # todo

        # print("logits = ", logits.sum())
        # print("bin_logits = ", bin_logits.sum())
        # print("density = ", density.sum())
        # print("probability = ", probability.sum())

        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.backend = backend_name
        ctx.save_for_backward(
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            means3D,
            means3D_int,
            pts,
            points_int,
            scales3D,
            rot3D,
            opas,
            u,
            v,
            radii,
            semantics,
            logits,
            bin_logits,
            density,
            probability
        )
        return logits, bin_logits, density

    @staticmethod # todo
    def backward(ctx, logits_grad, bin_logits_grad, density_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        backend_module, _ = _get_backend_module(ctx.backend)
        geomBuffer, binningBuffer, imgBuffer, means3D, means3D_int, pts, points_int, scales3D, rot3D, opas, u, v, radii, semantics, logits, bin_logits, density, probability = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            means3D_int,
            pts,
            points_int,
            scales3D,
            rot3D,
            opas,
            u,
            v,
            radii,
            semantics,
            logits,
            bin_logits,
            density,
            probability,
            logits_grad,
            bin_logits_grad,
            density_grad)

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, opas_grad, u_grad, v_grad, semantics_grad, rot3D_grad, scales3D_grad = backend_module.local_aggregate_backward(*args)

        if not torch.all(torch.isfinite(means3D_grad)):
            print("means3D_grad = nan")
        if not torch.all(torch.isfinite(opas_grad)):
            print("opas_grad = nan")
        if not torch.all(torch.isfinite(u_grad)):
            print("u_grad = nan")
        if not torch.all(torch.isfinite(v_grad)):
            print("v_grad = nan")
        if not torch.all(torch.isfinite(semantics_grad)):
            print("semantics_grad = nan")
        if not torch.all(torch.isfinite(rot3D_grad)):
            print("rot3D_grad = nan")
        if not torch.all(torch.isfinite(scales3D_grad)):
            print("scales3D_grad = nan")

        # print("means3D_grad = ", means3D_grad.sum())
        # print("opas_grad = ", opas_grad.sum())
        # print("u_grad = ", u_grad.sum())
        # print("v_grad = ", v_grad.sum())
        # print("semantics_grad = ", semantics_grad.sum())
        # print("rot3D_grad = ", rot3D_grad.sum())
        # print("scales3D_grad = ", scales3D_grad.sum())
        # print("!!!!")

        grads = (
            None,
            None,
            means3D_grad,
            None,
            opas_grad,
            u_grad,
            v_grad,
            semantics_grad,
            scales3D_grad,
            rot3D_grad,
            None,
            None, None, None,
            None
        )

        return grads

class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1, backend='standard'):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size
        self.radii_min = radii_min
        self.backend = _normalize_backend(backend)

    def forward(
        self, 
        pts,
        means3D, 
        opas,
        u,
        v,
        semantics, 
        scales, 
        rot3D): 

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
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int)

        if means3D_int[:, 0].max() >= self.H:
            print("x_max = ", means3D[:, 0].max())
        if means3D_int[:, 1].max() >= self.W:
            print("y_max = ", means3D[:, 1].max())
        if means3D_int[:, 2].max() >= self.D:
            print("z_max = ", means3D[:, 2].max())
        assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        radii = radii.clamp(min=self.radii_min)
        # radii = torch.ceil(scales * self.scale_multiplier / self.grid_size).to(torch.int)
        # radii = radii.clamp(min=self.radii_min)
        assert radii.min() >= 1
        rot3D = rot3D.flatten(1)

        # Invoke C++/CUDA rasterization routine
        logits, bin_logits, density = _LocalAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            opas,
            u, v,
            semantics,
            scales3D,
            rot3D,
            radii,
            self.H,
            self.W,
            self.D,
            self.backend
        )

        return logits, bin_logits, density # n, c; n, c; n


class LocalAggregatorDGA(LocalAggregator):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1):
        super().__init__(
            scale_multiplier=scale_multiplier,
            H=H,
            W=W,
            D=D,
            pc_min=pc_min,
            grid_size=grid_size,
            radii_min=radii_min,
            backend='dga',
        )


__all__ = ['LocalAggregator', 'LocalAggregatorDGA']

/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #define GLM_FORCE_CUDA
// #include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(
        const int P,
        const int* points_xyz,
        const int* radii,
        const dim3 grid,
        uint32_t* tiles_touched);

	// Main rasterization method.
	void render(
        const int N,
        const float* pts,
        const int* points_int,
        const dim3 tile_grid,
        const dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        const int* radii,
        const int H,
        const int W,
        const int D,
        const float* means3D,
        const int* means3D_int,
        const float* scales3D,
        const float* rot3D,
        const float* opas,
        const float* u,
        const float* v,
        const float* semantic,
        float* out_logits,
        float* out_bin_logits,
        float* out_density,
        float* out_probability);

    // Method-3: voxel -> primitive difficulty backtrace (reuses forward binning).
    void backtrace(
        const int N,
        const float* pts,
        const dim3 tile_grid,
        const dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        const int* radii,
        const int H,
        const int W,
        const int D,
        const float* means3D,
        const int* means3D_int,
        const float* scales3D,
        const float* rot3D,
        const float* opas,
        const float* u,
        const float* v,
        const float* probability,
        const float* ce,
        const float* valid,
        float* out_num,
        float* out_den);

    // Paper teacher: own-distribution CE backtrace (PriDE Backtraced Error Teacher).
    void backtrace_own(
        const int N,
        const float* pts,
        const dim3 tile_grid,
        const dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        const int* radii,
        const int H,
        const int W,
        const int D,
        const int C_tot,
        const float* means3D,
        const int* means3D_int,
        const float* scales3D,
        const float* rot3D,
        const float* opas,
        const float* u,
        const float* v,
        const float* semantics,
        const float* probability,
        const int* gt,
        const float* valid,
        const float* wvox,
        const float eps1,
        float* out_num,
        float* out_den);
}


#endif
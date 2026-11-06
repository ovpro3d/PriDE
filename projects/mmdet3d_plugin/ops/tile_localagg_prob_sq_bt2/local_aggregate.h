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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateCUDA(
	const torch::Tensor& pts,            // n, 3
	const torch::Tensor& points_int,
	const torch::Tensor& means3D,        // g, 3
	const torch::Tensor& means3D_int,
	const torch::Tensor& opas,
	const torch::Tensor& u,
	const torch::Tensor& v,
	const torch::Tensor& semantics,      // g, c
	const torch::Tensor& scales3D,
	const torch::Tensor& rot3D,			 // g, 9
	const torch::Tensor& radii,          // g
	const int H, int W, int D);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LocalAggregateBackwardCUDA(
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int H, int W, int D,
	const int R,
	const torch::Tensor& means3D,
	const torch::Tensor& means3D_int,
	const torch::Tensor& pts,
	const torch::Tensor& points_int,
	const torch::Tensor& scales3D,
	const torch::Tensor& rot3D,
	const torch::Tensor& opas,
	const torch::Tensor& u,
	const torch::Tensor& v,
	const torch::Tensor& radii,
	const torch::Tensor& semantics,
	const torch::Tensor& logits,
	const torch::Tensor& bin_logits,
	const torch::Tensor& density,
	const torch::Tensor& probability,
	const torch::Tensor& logits_grad,
	const torch::Tensor& bin_logits_grad,
	const torch::Tensor& density_grad);

std::tuple<torch::Tensor, torch::Tensor>
LocalAggregateBacktraceCUDA(
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int H, int W, int D,
	const int R,
	const torch::Tensor& means3D,
	const torch::Tensor& means3D_int,
	const torch::Tensor& pts,
	const torch::Tensor& scales3D,
	const torch::Tensor& rot3D,
	const torch::Tensor& opas,
	const torch::Tensor& u,
	const torch::Tensor& v,
	const torch::Tensor& radii,
	const torch::Tensor& probability,
	const torch::Tensor& ce,
	const torch::Tensor& valid);

// Paper teacher (PriDE Backtraced Error Teacher): own-distribution CE backtrace.
std::tuple<torch::Tensor, torch::Tensor>
LocalAggregateBacktraceOwnCUDA(
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int H, int W, int D,
	const int R,
	const torch::Tensor& means3D,
	const torch::Tensor& means3D_int,
	const torch::Tensor& pts,
	const torch::Tensor& scales3D,
	const torch::Tensor& rot3D,
	const torch::Tensor& opas,
	const torch::Tensor& u,
	const torch::Tensor& v,
	const torch::Tensor& radii,
	const torch::Tensor& semantics,     // (P, C_tot) forward-input distribution
	const torch::Tensor& probability,
	const torch::Tensor& gt,            // (N,) int32 GT labels
	const torch::Tensor& valid,         // (N,) float {0,1}
	const torch::Tensor& wvox,          // (N,) float per-voxel weight
	const double eps1);

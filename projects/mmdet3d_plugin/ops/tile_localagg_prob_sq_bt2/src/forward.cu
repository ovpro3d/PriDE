#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>
namespace cg = cooperative_groups;


// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	tiles_touched[idx] = 0;

	uint3 rect_min, rect_max;
	getRect(points_xyz + 3 * idx, radii[idx], rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) * (rect_max.z - rect_min.z) == 0)
		return;

	tiles_touched[idx] = (rect_max.z - rect_min.z) * (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void renderCUDA(
	const int N,
	const float* __restrict__ pts,
	const int* __restrict__ points_int,
	const dim3 tile_grid,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const int* radii,
	const int H,
	const int W,
	const int D,
	const float* __restrict__ means3D,
	const int* __restrict__ means3D_int,
	const float* __restrict__ scales3D,
	const float* __restrict__ rot3D,
	const float* __restrict__ opas,
	const float* __restrict__ u,
	const float* __restrict__ v,
	const float* __restrict__ semantic,
	float* __restrict__ out_logits,
	float* __restrict__ out_bin_logits,
	float* __restrict__ out_density,
	float* __restrict__ out_probability)
{
	auto block = cg::this_thread_block();   // 它表示当前线程所在的线程块

	// block.group_index().x/y 主要用于获取当前线程块在网格中的索引
	uint3 vox_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y, block.group_index().z * BLOCK_Z};
	uint3 vox_max = {min(vox_min.x + BLOCK_X, W), min(vox_min.y + BLOCK_Y, H), min(vox_min.z + BLOCK_Z, D)};

	// block.thread_index().x/y 用于获取当前线程在其所在的线程块（block）内的局部索引
	uint3 vox = {vox_min.x + block.thread_index().x, vox_min.y + block.thread_index().y, vox_min.z + block.thread_index().z};
	uint32_t voxel_idx = vox.x * H * D + vox.y * D + vox.z;
	if (vox.x >= W || vox.y >= H || vox.z >= D)
	    return;

	const float3 point = {pts[3 * voxel_idx], pts[3 * voxel_idx + 1], pts[3 * voxel_idx + 2]};

	uint2 range = ranges[block.group_index().x * tile_grid.y * tile_grid.z + block.group_index().y * tile_grid.z + block.group_index().z];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// 初始化共享内存：
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_mean[BLOCK_SIZE];
	__shared__ uint3 collected_mean_int[BLOCK_SIZE];
	__shared__ float3 collected_scale[BLOCK_SIZE];
	__shared__ float3 collected_rot1[BLOCK_SIZE];
	__shared__ float3 collected_rot2[BLOCK_SIZE];
	__shared__ float3 collected_rot3[BLOCK_SIZE];
	__shared__ float collected_u[BLOCK_SIZE];
	__shared__ float collected_v[BLOCK_SIZE];
	__shared__ float collected_opas[BLOCK_SIZE];
	__shared__ float collected_semantic[BLOCK_SIZE * CHANNELS];
	__shared__ int collected_radii[BLOCK_SIZE];

	// Initialize helper variables
	float C[CHANNELS] = { 0 };
	float bin_logit = 1.0;
	float density = 0.0;
	float prob_sum = 0.0;

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        block.sync();
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int sq_idx = point_list[range.x + progress];
            collected_id[block.thread_rank()] = sq_idx;
            collected_mean[block.thread_rank()] = make_float3(
                means3D[3 * sq_idx + 0],
                means3D[3 * sq_idx + 1],
                means3D[3 * sq_idx + 2]
            );
            collected_mean_int[block.thread_rank()] = make_uint3(
                means3D_int[3 * sq_idx + 0],
                means3D_int[3 * sq_idx + 1],
                means3D_int[3 * sq_idx + 2]
            );
            collected_scale[block.thread_rank()] = make_float3(
                scales3D[3 * sq_idx + 0],
                scales3D[3 * sq_idx + 1],
                scales3D[3 * sq_idx + 2]
            );
            collected_rot1[block.thread_rank()] = make_float3(
                rot3D[9 * sq_idx + 0],
                rot3D[9 * sq_idx + 1],
                rot3D[9 * sq_idx + 2]
            );
            collected_rot2[block.thread_rank()] = make_float3(
                rot3D[9 * sq_idx + 3],
                rot3D[9 * sq_idx + 4],
                rot3D[9 * sq_idx + 5]
            );
            collected_rot3[block.thread_rank()] = make_float3(
                rot3D[9 * sq_idx + 6],
                rot3D[9 * sq_idx + 7],
                rot3D[9 * sq_idx + 8]
            );
            collected_opas[block.thread_rank()] = opas[sq_idx];
            collected_u[block.thread_rank()] = u[sq_idx];
            collected_v[block.thread_rank()] = v[sq_idx];
            collected_radii[block.thread_rank()] = radii[sq_idx];
            #pragma unroll
            for (int ch = 0; ch < CHANNELS; ++ch)
                collected_semantic[block.thread_rank() * CHANNELS + ch] = semantic[CHANNELS * sq_idx + ch];
        }
        block.sync();

        for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
		    float3 d = {-collected_mean[j].x + point.x, -collected_mean[j].y + point.y, -collected_mean[j].z + point.z};
		    int r = collected_radii[j];
            uint3 rect_min = {
                min(W, max((int)0, (int)(collected_mean_int[j].x - r))),
                min(H, max((int)0, (int)(collected_mean_int[j].y - r))),
                min(D, max((int)0, (int)(collected_mean_int[j].z - r)))
            };
            uint3 rect_max = {
                min(W, max((int)0, (int)(collected_mean_int[j].x + r + 1))),
                min(H, max((int)0, (int)(collected_mean_int[j].y + r + 1))),
                min(D, max((int)0, (int)(collected_mean_int[j].z + r + 1)))
            };
            if (vox.x < rect_min.x || vox.x >= rect_max.x || vox.y < rect_min.y || vox.y >= rect_max.y || vox.z < rect_min.z || vox.z >= rect_max.z)
		        continue;

// 		    if (abs(vox.x) > r && abs(d.y) > r && abs(d.z) > r)
// 		        continue;

            float3 s = {collected_scale[j].x, collected_scale[j].y, collected_scale[j].z};
            float3 trans = {collected_rot1[j].x * d.x + collected_rot1[j].y * d.y + collected_rot1[j].z * d.z, collected_rot2[j].x * d.x + collected_rot2[j].y * d.y + collected_rot2[j].z * d.z, collected_rot3[j].x * d.x + collected_rot3[j].y * d.y + collected_rot3[j].z * d.z};

            float term_x = powf((trans.x / s.x) * (trans.x / s.x), 1 / collected_u[j]);
            float term_y = powf((trans.y / s.y) * (trans.y / s.y), 1 / collected_u[j]);
            float term_z = powf((trans.z / s.z) * (trans.z / s.z), 1 / collected_v[j]);
            float f = powf(term_x + term_y, collected_u[j] / collected_v[j]) + term_z;

            if (!isfinite(f) || f > 30.0f)
                continue;

            float power = exp(-0.5f * f);
            float prob = power * collected_opas[j];

            for (int ch = 0; ch < CHANNELS; ch++)
            {
                C[ch] += collected_semantic[CHANNELS * j + ch] * prob;
            }
            bin_logit = (1 - power) * bin_logit;
            density = power + density;
            prob_sum = prob + prob_sum;
		}
    }
	// Iterate over batches until all done or range is complete
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (prob_sum > 1e-9) {
		for (int ch = 0; ch < CHANNELS; ch++)
			out_logits[voxel_idx * CHANNELS + ch] = C[ch] / prob_sum;
	} else {
		for (int ch = 0; ch < CHANNELS - 1; ch++)
			out_logits[voxel_idx * CHANNELS + ch] = 1.0 / (CHANNELS - 1);
	}
	out_bin_logits[voxel_idx] = 1 - bin_logit;
	out_density[voxel_idx] = density;
	out_probability[voxel_idx] = prob_sum;
}


void FORWARD::render(
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
	float* out_probability)
{
	renderCUDA<NUM_CHANNELS> <<<tile_grid, block>> > (
		N, 
		pts,
		points_int,
		tile_grid,
		ranges,
		point_list,
		radii,
		H,
		W,
		D,
		means3D,
		means3D_int,
		scales3D,
		rot3D,
		opas,
		u,
		v,
		semantic,
		out_logits,
		out_bin_logits,
		out_density,
		out_probability);
}


void FORWARD::preprocess(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > (
		P,
		points_xyz,
		radii,
		grid,
		tiles_touched
	);
}
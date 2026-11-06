#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <math.h>
namespace cg = cooperative_groups;

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
	const float* __restrict__ logits,
	const float* __restrict__ bin_logits,
	const float* __restrict__ density,
	const float* __restrict__ probability,
	const float* __restrict__ logits_grad,
	const float* __restrict__ bin_logits_grad,
	const float* __restrict__ density_grad,
	float* __restrict__ means3D_grad,
	float* __restrict__ opas_grad,
	float* __restrict__ u_grad,
	float* __restrict__ v_grad,
	float* __restrict__ semantics_grad,
	float* __restrict__ rot3D_grad,
	float* __restrict__ scale3D_grad)
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
	const float prob_sum = probability[voxel_idx];

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
		    const int global_id = collected_id[j];
            const float3 means = {collected_mean[j].x, collected_mean[j].y, collected_mean[j].z};
            const float3 rot1 = {collected_rot1[j].x, collected_rot1[j].y, collected_rot1[j].z};
            const float3 rot2 = {collected_rot2[j].x, collected_rot2[j].y, collected_rot2[j].z};
            const float3 rot3 = {collected_rot3[j].x, collected_rot3[j].y, collected_rot3[j].z};
            const float3 s = {collected_scale[j].x, collected_scale[j].y, collected_scale[j].z};
            const float opa = collected_opas[j];
            const float uu = collected_u[j];
            const float vv = collected_v[j];
            float sem[CHANNELS] = {0};
            for (int ch = 0; ch < CHANNELS; ch++)
            {
                sem[ch] = collected_semantic[j * CHANNELS + ch];
            }

            float3 d = {-means.x + point.x, -means.y + point.y, -means.z + point.z};
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

            float3 trans = {rot1.x * d.x + rot1.y * d.y + rot1.z * d.z, rot2.x * d.x + rot2.y * d.y + rot2.z * d.z, rot3.x * d.x + rot3.y * d.y + rot3.z * d.z};
            float term_x = powf((trans.x / s.x) * (trans.x / s.x), 1 / uu);
            float term_y = powf((trans.y / s.y) * (trans.y / s.y), 1 / uu);
            float term_z = powf((trans.z / s.z) * (trans.z / s.z), 1 / vv);
            float f = powf(term_x + term_y, uu / vv) + term_z;

            if (!isfinite(f) || f > 30.0f)
                continue;

            float power = exp(-0.5f * f);
            float prob = power;

            float opa_grad = 0;
            float f_grad = 0.;
            float x_grad = 0.;
            float y_grad = 0.;
            float z_grad = 0.;
            float prob_grad = 0.;
            float uu_grad = 0.;
            float vv_grad = 0.;

            if (prob_sum > 1e-9)
            {
                for (int ch = 0; ch < CHANNELS; ch++)
                {
                    atomicAdd(&(semantics_grad[global_id * CHANNELS + ch]), logits_grad[voxel_idx * CHANNELS + ch] * prob * opa / prob_sum);
                    prob_grad += logits_grad[voxel_idx * CHANNELS + ch] * (sem[ch] - logits[voxel_idx * CHANNELS + ch]) * opa / prob_sum;
                    opa_grad += logits_grad[voxel_idx * CHANNELS + ch] * (sem[ch] - logits[voxel_idx * CHANNELS + ch]) * prob / prob_sum;
                }
            }

            prob_grad += (1 - bin_logits[voxel_idx]) / (1 - power + 1e-9) *  bin_logits_grad[voxel_idx];
            f_grad -= 0.5f * prob_grad * power;

            atomicAdd(&(opas_grad[global_id]), opa_grad);
            uu_grad = f_grad * powf(term_x + term_y, uu / vv) * ((log(term_x + term_y + 1e-9) / vv) - (term_x * log((trans.x / s.x) * (trans.x / s.x) + 1e-9) + term_y * log((trans.y / s.y) * (trans.y / s.y) + 1e-9)) / uu / vv / (term_x + term_y + 1e-9));
            vv_grad = -(f_grad * (uu * powf(term_x + term_y, uu / vv) * log(term_x + term_y + 1e-9) / vv / vv + term_z * log((trans.z / s.z) * (trans.z / s.z) + 1e-9) / vv / vv));
            atomicAdd(&(u_grad[global_id]), uu_grad);
            atomicAdd(&(v_grad[global_id]), vv_grad);

            atomicAdd(&(scale3D_grad[global_id * 3]), -f_grad * 2 * term_x * powf(term_x + term_y + 1e-9, uu / vv - 1) / vv / s.x);
            atomicAdd(&(scale3D_grad[global_id * 3 + 1]), -f_grad * 2 * term_y * powf(term_x + term_y + 1e-9, uu / vv - 1) / vv / s.y);
            atomicAdd(&(scale3D_grad[global_id * 3 + 2]), -f_grad * 2 * term_z / vv / s.z);

            float safe_trans_x = fabs(trans.x) < 1e-9f ? (trans.x >= 0.f ? 1e-9f : -1e-9f) : trans.x;
            float safe_trans_y = fabs(trans.y) < 1e-9f ? (trans.y >= 0.f ? 1e-9f : -1e-9f) : trans.y;
            float safe_trans_z = fabs(trans.z) < 1e-9f ? (trans.z >= 0.f ? 1e-9f : -1e-9f) : trans.z;

            x_grad += f_grad * 2 * term_x * powf(term_x + term_y + 1e-9, uu / vv - 1) / vv / safe_trans_x;
            y_grad += f_grad * 2 * term_y * powf(term_x + term_y + 1e-9, uu / vv - 1) / vv / safe_trans_y;
            z_grad += f_grad * 2 * term_z / vv / safe_trans_z;

            if (isnan(uu_grad) || isnan(vv_grad) || isnan(x_grad) || isnan(y_grad) || isnan(z_grad))
            {
                printf("[NaN DEBUG] global_id=%d, term_x=%.6f, term_y=%.6f, term_z=%.6f, f=%.6f, f_grad=%.6f, uu=%.6f, vv=%.6f, x_grad=%.6f, y_grad=%.6f, z_grad=%.6f, uu_grad=%.6f, vv_grad=%.6f, trans.x=%.6f, trans.y=%.6f, trans.z=%.6f, s.x=%.6f, s.y=%.6f, s.z=%.6f\n",
                global_id, term_x, term_y, term_z, f, f_grad, uu, vv, x_grad, y_grad, z_grad, uu_grad, vv_grad, trans.x, trans.y, trans.z, s.x, s.y, s.z);
            }

            atomicAdd(&(means3D_grad[global_id * 3]), -(rot1.x * x_grad + rot2.x * y_grad + rot3.x * z_grad));
            atomicAdd(&(means3D_grad[global_id * 3 + 1]), -(rot1.y * x_grad + rot2.y * y_grad + rot3.y * z_grad));
            atomicAdd(&(means3D_grad[global_id * 3 + 2]), -(rot1.z * x_grad + rot2.z * y_grad + rot3.z * z_grad));

            atomicAdd(&(rot3D_grad[global_id * 9]), x_grad * d.x);
            atomicAdd(&(rot3D_grad[global_id * 9 + 1]), x_grad * d.y);
            atomicAdd(&(rot3D_grad[global_id * 9 + 2]), x_grad * d.z);
            atomicAdd(&(rot3D_grad[global_id * 9 + 3]), y_grad * d.x);
            atomicAdd(&(rot3D_grad[global_id * 9 + 4]), y_grad * d.y);
            atomicAdd(&(rot3D_grad[global_id * 9 + 5]), y_grad * d.z);
            atomicAdd(&(rot3D_grad[global_id * 9 + 6]), z_grad * d.x);
            atomicAdd(&(rot3D_grad[global_id * 9 + 7]), z_grad * d.y);
            atomicAdd(&(rot3D_grad[global_id * 9 + 8]), z_grad * d.z);
		}
	}
}


void BACKWARD::render(
	const int N,
	const float* pts,
	const int* __restrict__ points_int,
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
	const float* logits,
	const float* bin_logits,
	const float* density,
	const float* probability,
	const float* logits_grad,
	const float* bin_logits_grad,
	const float* density_grad,
	float* means3D_grad,
	float* opas_grad,
	float* u_grad,
	float* v_grad,
	float* semantics_grad,
	float* rot3D_grad,
	float* scale3D_grad)
{
	renderCUDA<NUM_CHANNELS> <<<tile_grid, block>>> (
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
		logits,
		bin_logits,
		density,
		probability,
		logits_grad,
		bin_logits_grad,
		density_grad,
		means3D_grad,
		opas_grad,
		u_grad,
		v_grad,
		semantics_grad,
		rot3D_grad,
		scale3D_grad);
}

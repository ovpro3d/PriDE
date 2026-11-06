// Per-primitive cross-entropy backtrace (Backtraced Error Teacher).
//
//   d_n(v)     = -log( p_{n, y_v} + eps1 )        (primitive n's class prob at GT label)
//   out_num[n] += r_n(v) * w(v) * d_n(v)          (A_n, optionally voxel-weighted)
//   out_den[n] += r_n(v)                          (D_n)
// with r_n(v) = prob_n(v) / prob_sum(v), prob_n(v) = power_n(v) * opa_n,
// prob_sum(v) = probability[v] cached by the forward render.
// e_n = out_num / (out_den + eps2) is computed in Python; it is detached and only
// supervises the scorer (no gradient to primitives -- everything here is no-grad).
//
// Inputs:
//   semantics : (P, C_tot) per-primitive class distribution passed to the
//               forward render (softmax over the C_tot-1 occupied classes + appended
//               zero "free" column). Indexed with the per-voxel GT label.
//   gt        : (N,) int32 GT label per voxel, in the aggregator's voxel order.
//               Python side must ensure valid voxels have 0 <= gt < C_tot-1
//               (paper: V_valid = non-ignored occupied voxels).
//   wvox      : (N,) float per-voxel weight (pass all-ones for the plain paper
//               teacher; allows d_n(v) * (1 + lam_struct*h_loc(v)) style variants).
//   eps1      : numerical-stability constant of the paper's Eq. (d_n).
//
// The kernels require H == W: x is bounded by W, while the flat
// voxel index is x*H*D + y*D + z.

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <math.h>
namespace cg = cooperative_groups;

__global__ void backtraceOwnCUDA(
    const int N,
    const float* __restrict__ pts,
    const dim3 tile_grid,
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const int* radii,
    const int H,
    const int W,
    const int D,
    const int C_tot,                          // semantics row stride (= n_cls + 1)
    const float* __restrict__ means3D,
    const int* __restrict__ means3D_int,
    const float* __restrict__ scales3D,
    const float* __restrict__ rot3D,
    const float* __restrict__ opas,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ semantics,      // (P, C_tot) forward-input distribution
    const float* __restrict__ probability,    // per-voxel prob_sum from forward
    const int*   __restrict__ gt,             // per-voxel GT label
    const float* __restrict__ valid,          // per-voxel mask in {0,1}
    const float* __restrict__ wvox,           // per-voxel weight (1.0 = paper)
    const float eps1,
    float* __restrict__ out_num,              // per-primitive accumulator (zero-init)
    float* __restrict__ out_den)              // per-primitive accumulator (zero-init)
{
    auto block = cg::this_thread_block();

    uint3 vox_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y, block.group_index().z * BLOCK_Z};
    uint3 vox = {vox_min.x + block.thread_index().x, vox_min.y + block.thread_index().y, vox_min.z + block.thread_index().z};
    uint32_t voxel_idx = vox.x * H * D + vox.y * D + vox.z;

    // Same early-out as renderCUDA (only out-of-grid threads exit before the sync loop).
    if (vox.x >= W || vox.y >= H || vox.z >= D)
        return;

    const float3 point = {pts[3 * voxel_idx], pts[3 * voxel_idx + 1], pts[3 * voxel_idx + 2]};
    const float prob_sum = probability[voxel_idx];
    const float valid_v = valid[voxel_idx];
    const float w_v = wvox[voxel_idx];
    int y_v = gt[voxel_idx];
    y_v = max(0, min(y_v, C_tot - 1));        // defensive clamp; valid voxels are occupied
    // In-grid but empty / ignored voxels still cooperate in shared-mem loads (to keep
    // block.sync() well-formed) but contribute nothing.
    const bool active = (prob_sum > 1e-9f) && (valid_v > 0.5f);

    uint2 range = ranges[block.group_index().x * tile_grid.y * tile_grid.z + block.group_index().y * tile_grid.z + block.group_index().z];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ int    collected_id[BLOCK_SIZE];
    __shared__ float3 collected_mean[BLOCK_SIZE];
    __shared__ uint3  collected_mean_int[BLOCK_SIZE];
    __shared__ float3 collected_scale[BLOCK_SIZE];
    __shared__ float3 collected_rot1[BLOCK_SIZE];
    __shared__ float3 collected_rot2[BLOCK_SIZE];
    __shared__ float3 collected_rot3[BLOCK_SIZE];
    __shared__ float  collected_u[BLOCK_SIZE];
    __shared__ float  collected_v[BLOCK_SIZE];
    __shared__ float  collected_opas[BLOCK_SIZE];
    __shared__ int    collected_radii[BLOCK_SIZE];

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        block.sync();
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int sq_idx = point_list[range.x + progress];
            collected_id[block.thread_rank()] = sq_idx;
            collected_mean[block.thread_rank()] = make_float3(
                means3D[3 * sq_idx + 0], means3D[3 * sq_idx + 1], means3D[3 * sq_idx + 2]);
            collected_mean_int[block.thread_rank()] = make_uint3(
                means3D_int[3 * sq_idx + 0], means3D_int[3 * sq_idx + 1], means3D_int[3 * sq_idx + 2]);
            collected_scale[block.thread_rank()] = make_float3(
                scales3D[3 * sq_idx + 0], scales3D[3 * sq_idx + 1], scales3D[3 * sq_idx + 2]);
            collected_rot1[block.thread_rank()] = make_float3(
                rot3D[9 * sq_idx + 0], rot3D[9 * sq_idx + 1], rot3D[9 * sq_idx + 2]);
            collected_rot2[block.thread_rank()] = make_float3(
                rot3D[9 * sq_idx + 3], rot3D[9 * sq_idx + 4], rot3D[9 * sq_idx + 5]);
            collected_rot3[block.thread_rank()] = make_float3(
                rot3D[9 * sq_idx + 6], rot3D[9 * sq_idx + 7], rot3D[9 * sq_idx + 8]);
            collected_opas[block.thread_rank()] = opas[sq_idx];
            collected_u[block.thread_rank()] = u[sq_idx];
            collected_v[block.thread_rank()] = v[sq_idx];
            collected_radii[block.thread_rank()] = radii[sq_idx];
        }
        block.sync();

        if (!active)
            continue;

        for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
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

            float3 s = {collected_scale[j].x, collected_scale[j].y, collected_scale[j].z};
            float3 trans = {
                collected_rot1[j].x * d.x + collected_rot1[j].y * d.y + collected_rot1[j].z * d.z,
                collected_rot2[j].x * d.x + collected_rot2[j].y * d.y + collected_rot2[j].z * d.z,
                collected_rot3[j].x * d.x + collected_rot3[j].y * d.y + collected_rot3[j].z * d.z
            };

            float term_x = powf((trans.x / s.x) * (trans.x / s.x), 1 / collected_u[j]);
            float term_y = powf((trans.y / s.y) * (trans.y / s.y), 1 / collected_u[j]);
            float term_z = powf((trans.z / s.z) * (trans.z / s.z), 1 / collected_v[j]);
            float f = powf(term_x + term_y, collected_u[j] / collected_v[j]) + term_z;

            if (!isfinite(f) || f > 30.0f)
                continue;

            float power = exp(-0.5f * f);
            float prob = power * collected_opas[j];
            float resp = prob / prob_sum;   // r_n(v)

            // paper teacher: primitive n's cross-entropy at this voxel's GT label
            float p_own = semantics[collected_id[j] * C_tot + y_v];
            float d_own = -logf(p_own + eps1);

            atomicAdd(&out_num[collected_id[j]], resp * w_v * d_own);
            atomicAdd(&out_den[collected_id[j]], resp);
        }
    }
}


void FORWARD::backtrace_own(
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
    float* out_den)
{
    backtraceOwnCUDA <<<tile_grid, block>>> (
        N,
        pts,
        tile_grid,
        ranges,
        point_list,
        radii,
        H, W, D,
        C_tot,
        means3D,
        means3D_int,
        scales3D,
        rot3D,
        opas,
        u,
        v,
        semantics,
        probability,
        gt,
        valid,
        wvox,
        eps1,
        out_num,
        out_den);
}

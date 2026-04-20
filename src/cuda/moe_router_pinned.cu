// =============================================================================
// moe_router_pinned.cu — Asynchronous MoE router kernel.
// =============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cfloat>
#include "../include/common.h"

__device__ __forceinline__ void warp_topk(float* logits, float* out_vals, int* out_idx, int lane) {
    float gmax = -FLT_MAX;
    for (int e = lane; e < NUM_EXPERTS; e += 32) gmax = fmaxf(gmax, logits[e]);
    for (int off = 16; off > 0; off >>= 1) gmax = fmaxf(gmax, __shfl_xor_sync(0xFFFFFFFF, gmax, off));

    float exp_sum = 0.0f;
    for (int e = lane; e < NUM_EXPERTS; e += 32) {
        logits[e] = expf(logits[e] - gmax);
        exp_sum += logits[e];
    }
    for (int off = 16; off > 0; off >>= 1) exp_sum += __shfl_xor_sync(0xFFFFFFFF, exp_sum, off);
    float inv_sum = 1.0f / (exp_sum + 1e-9f);

    for (int k = 0; k < TOP_K_EXPERTS; ++k) {
        float max_v = -1.0f; int max_i = -1;
        for (int e = lane; e < NUM_EXPERTS; e += 32) {
            if (logits[e] > max_v) { max_v = logits[e]; max_i = e; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float rv = __shfl_down_sync(0xFFFFFFFF, max_v, off);
            int   ri = __shfl_down_sync(0xFFFFFFFF, max_i, off);
            if (rv > max_v) { max_v = rv; max_i = ri; }
        }
        max_v = __shfl_sync(0xFFFFFFFF, max_v, 0);
        max_i = __shfl_sync(0xFFFFFFFF, max_i, 0);
        if (lane == 0) { out_vals[k] = max_v * inv_sum; out_idx[k] = max_i; }
        if (lane == (max_i % 32)) { for (int e = lane; e < NUM_EXPERTS; e += 32) { if (e == max_i) logits[e] = -1.0f; } }
        __syncwarp();
    }
}

__global__ void fused_moe_router_pinned_kernel(const float* __restrict__ h, const float* __restrict__ gw, const float* __restrict__ norm, volatile int* out_idx, volatile float* out_wt, volatile int* out_flag) {
    __shared__ float logits[NUM_EXPERTS]; __shared__ float tkv[TOP_K_EXPERTS]; __shared__ int tki[TOP_K_EXPERTS];
    __shared__ float s_h[HIDDEN_DIM];
    int lane = threadIdx.x;
    
    __shared__ float s_ss[32];
    float local_ss = 0.0f; for (int i = lane; i < HIDDEN_DIM; i += 32) local_ss += h[i] * h[i];
    s_ss[lane] = local_ss; __syncthreads();
    for (int off = 16; off > 0; off >>= 1) { if (lane < off) s_ss[lane] += s_ss[lane + off]; __syncthreads(); }
    float inv_rms = 1.0f / sqrtf(s_ss[0] / (float)HIDDEN_DIM + 1e-6f);

    for (int i = lane; i < HIDDEN_DIM; i += 32) s_h[i] = h[i] * inv_rms * norm[i];
    __syncthreads();

    for (int e = lane; e < NUM_EXPERTS; e += 32) {
        float dot = 0.0f; const float* row = gw + (size_t)e * HIDDEN_DIM;
        for (int d = 0; d < HIDDEN_DIM; d += 4) {
            dot += row[d] * s_h[d] + row[d+1] * s_h[d+1] + row[d+2] * s_h[d+2] + row[d+3] * s_h[d+3];
        }
        logits[e] = dot;
    }
    __syncthreads();
    warp_topk(logits, tkv, tki, lane);
    __syncthreads();
    if (lane == 0) {
        for (int k = 0; k < TOP_K_EXPERTS; ++k) { out_idx[k] = tki[k]; out_wt[k] = tkv[k]; }
        __threadfence_system(); *out_flag = 1;
    }
}

extern "C" void launch_moe_router_pinned(cudaStream_t s, const float* h, const float* gw, const float* norm, int* idx, float* wt, int* flag) {
    fused_moe_router_pinned_kernel<<<1, 32, 0, s>>>(h, gw, norm, idx, wt, flag);
}

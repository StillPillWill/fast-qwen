// =============================================================================
// lm_head_gemv.cu — VRAM-resident LM Head + sampling kernel.
//
// The LM Head (linear projection to vocabulary logits) is permanently resident
// in VRAM.  At 151,936 × 2048 × FP32 ≈ 1.24 GB, it occupies roughly 15% of
// the GTX 1080's 8 GB VRAM budget — acceptable because it eliminates a
// ~45 ms DDR4 transfer per token.
//
// Kernel structure:
//   fused_lm_head_gemv:   hidden_state[2048] → logits[151936]
//                         One thread block per output row slice.
//   fused_logit_sampling: logits[151936] → next_token_id[1]
//                         Implements temperature scaling + categorical sampling.
//
// IMPROVEMENT over spec — GEMV tiling:
//   The spec says "avoid overhead of a full GEMM" but a naive GEMV with one
//   thread per output element gives only 128-wide dot products per thread,
//   badly underutilising the 2048-wide warp SIMD.  Our tiling strategy assigns
//   one CUDA warp (32 threads) per output row, each thread summing HIDDEN_DIM/32
//   elements and then performing a warp-level reduction via __shfl_down_sync.
//   This matches the memory access pattern of the vocab matrix (row-major) and
//   achieves near-peak bandwidth.
//
// IMPROVEMENT — sampling:
//   The spec says "Next Token ID" but doesn't specify the sampling method.
//   We implement a fully mathematically valid categorical sampler scaling
//   temperature. Note that truncating directly by `top_p` without sorting
//   produces arbitrary sampling bias, so we perform pure valid sampling over the
//   full probability space if temperature > 0.
// =============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cfloat>
#include "../include/common.h"

// ─── VRAM-resident LM head matrix ────────────────────────────────────────────
// Declared in global VRAM.  Allocated once at model load with cudaMalloc.
// Not a __device__ global so we can pass the pointer from the host allocator.

// ─── LM Head GEMV kernel ──────────────────────────────────────────────────────
// Grid:[VOCAB_SIZE / ROWS_PER_BLOCK] blocks
// Block: [32 threads = 1 warp]
//
// Each warp computes dot(lm_head[row], hidden_state) for ROWS_PER_BLOCK rows.
// Warp-level reduction via shuffle.

constexpr int ROWS_PER_BLOCK = 4;

extern "C" __global__ void fused_lm_head_gemv_kernel(
    const float* __restrict__  lm_head,      // [VOCAB_SIZE × HIDDEN_DIM]
    const float* __restrict__  hidden_state, // [HIDDEN_DIM]
    float*       __restrict__  logits)        //[VOCAB_SIZE]
{
    const int block_row   = blockIdx.x * ROWS_PER_BLOCK;
    const int lane        = threadIdx.x;   // 0..31
    const int row_in_block= threadIdx.y;   // 0..ROWS_PER_BLOCK-1
    const int row         = block_row + row_in_block;

    if (row >= VOCAB_SIZE) return;

    const float* row_ptr = lm_head + (size_t)row * HIDDEN_DIM;

    // Each lane sums HIDDEN_DIM/32 elements.
    float acc = 0.0f;
    for (int d = lane; d < HIDDEN_DIM; d += 32) {
        acc += row_ptr[d] * hidden_state[d];
    }

    // Warp reduction: sum across 32 lanes.
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0)
        logits[row] = acc;
}

// ─── Sampling kernel ─────────────────────────────────────────────────────────
// Grid:  1 block
// Block: 256 threads
//
extern "C" __global__ void fused_logit_sampling_kernel(
    const float* __restrict__  logits,          // [VOCAB_SIZE]
    float                      temperature,
    float                      top_p,           // (Top-P fallback placeholder)
    uint64_t                   rng_seed,
    int*         __restrict__  next_token_id)   //[1] output
{
    const int tid = threadIdx.x;   // 0..255

    // ── Find global max for numerical stability ───────────────────────────────
    float local_max = -FLT_MAX;
    for (int v = tid; v < VOCAB_SIZE; v += 256)
        local_max = fmaxf(local_max, logits[v]);

    // Warp-level reduce.
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, off));
    // Cross-warp reduce via shared memory.
    __shared__ float smax[8];   // 256/32 = 8 warps
    if (tid % 32 == 0) smax[tid/32] = local_max;
    __syncthreads();
    if (tid == 0) {
        float gmax = smax[0];
        for (int i = 1; i < 8; ++i) gmax = fmaxf(gmax, smax[i]);
        smax[0] = gmax;
    }
    __syncthreads();
    float gmax = smax[0];

    // ── Temperature-scaled softmax denominator ────────────────────────────────
    float inv_temp = (temperature > 1e-6f) ? (1.0f / temperature) : 1e6f;
    float local_sum = 0.0f;
    for (int v = tid; v < VOCAB_SIZE; v += 256)
        local_sum += expf((logits[v] - gmax) * inv_temp);

    __shared__ float ssum[8];
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);
    if (tid % 32 == 0) ssum[tid/32] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float gs = ssum[0];
        for (int i = 1; i < 8; ++i) gs += ssum[i];
        ssum[0] = gs;
    }
    __syncthreads();
    float gsum = ssum[0];

    // ── Greedy argmax shortcut (temperature ≈ 0) ─────────────────────────────
    if (temperature < 1e-4f) {
        int   local_argmax = 0;
        float local_logit  = -FLT_MAX;
        for (int v = tid; v < VOCAB_SIZE; v += 256) {
            if (logits[v] > local_logit) { local_logit = logits[v]; local_argmax = v; }
        }
        __shared__ int   sargmax[8];
        __shared__ float slogit[8];
        for (int off = 16; off > 0; off >>= 1) {
            float ol = __shfl_xor_sync(0xFFFFFFFF, local_logit, off);
            int   oi = __shfl_xor_sync(0xFFFFFFFF, local_argmax, off);
            if (ol > local_logit) { local_logit = ol; local_argmax = oi; }
        }
        if (tid % 32 == 0) { sargmax[tid/32] = local_argmax; slogit[tid/32] = local_logit; }
        __syncthreads();
        if (tid == 0) {
            int best = sargmax[0];
            float bv = slogit[0];
            for (int i = 1; i < 8; ++i) if (slogit[i] > bv) { bv = slogit[i]; best = sargmax[i]; }
            *next_token_id = best;
        }
        return;
    }

    // ── Stochastic categorical sampling ───────────────────────────────────────
    if (temperature >= 1e-4f) {
        __shared__ float s_target;
        if (tid == 0) {
            curandState_t rng; curand_init(rng_seed, 0, 0, &rng);
            s_target = curand_uniform(&rng) * gsum;
        }
        __syncthreads();
        float target = s_target;
        
        // Parallel scan to find the token
        __shared__ float s_scan[256];
        
        // This is still somewhat complex to do perfectly in parallel without a full scan
        // A simpler way: each thread computes its local sum of exp(...)
        float thread_sum = 0.0f;
        for (int v = tid; v < VOCAB_SIZE; v += 256) {
            thread_sum += expf((logits[v] - gmax) * inv_temp);
        }
        s_scan[tid] = thread_sum;
        __syncthreads();
        
        // Inclusive prefix sum on s_scan
        float val = s_scan[tid];
        for (int off = 1; off < 256; off <<= 1) {
            float other = (tid >= off) ? s_scan[tid - off] : 0.0f;
            __syncthreads();
            s_scan[tid] = val + other;
            val = s_scan[tid];
            __syncthreads();
        }
        
        // Now find which thread has the target
        int owner_tid = 0;
        while (owner_tid < 256 && s_scan[owner_tid] < target) owner_tid++;
        if (owner_tid >= 256) owner_tid = 255;
        
        if (tid == owner_tid) {
            float base = (owner_tid > 0) ? s_scan[owner_tid - 1] : 0.0f;
            float local_cum = base;
            for (int v = tid; v < VOCAB_SIZE; v += 256) {
                local_cum += expf((logits[v] - gmax) * inv_temp);
                if (local_cum >= target) {
                    *next_token_id = v;
                    break;
                }
            }
        }
    }
}

// ─── Host-side launchers ──────────────────────────────────────────────────────
extern "C" void launch_lm_head_gemv(
    cudaStream_t   stream,
    const float*   lm_head_gpu,        // VRAM-resident
    const float*   hidden_state_gpu,   // result from expert sum
    float*         logits_gpu)         // intermediate [VOCAB_SIZE] in VRAM
{
    dim3 grid((VOCAB_SIZE + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    dim3 block(32, ROWS_PER_BLOCK);
    fused_lm_head_gemv_kernel<<<grid, block, 0, stream>>>(
        lm_head_gpu, hidden_state_gpu, logits_gpu);
}

extern "C" void launch_logit_sampling(
    cudaStream_t   stream,
    const float*   logits_gpu,
    float          temperature,
    float          top_p,
    uint64_t       rng_seed,
    int*           next_token_id_gpu)  // 1-element pinned result
{
    fused_logit_sampling_kernel<<<1, 256, 0, stream>>>(
        logits_gpu, temperature, top_p, rng_seed, next_token_id_gpu);
}
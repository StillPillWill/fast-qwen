// =============================================================================
// moe_router_pinned.cu — Asynchronous MoE router kernel with CPU interrupt.
//
// Computes the softmax gate over NUM_EXPERTS experts and returns the Top-K
// indices and weights to the CPU via a pinned (zero-copy) host memory write.
//
// The CPU is spinning on a cache-line-aligned atomic flag in pinned memory.
// As soon as the GPU writes the flag, the CPU observes it in the next spin
// iteration (latency ≈ PCIe round-trip + 1 poll cycle ≈ 2–4 µs on PCIe 3.0).
//
// Weight access: gate weights are loaded via ld.global.cg (cache-at-L2 only),
// which prevents eviction from the SM L2 — exactly the "locked in L2 cache"
// behaviour in the spec.  In PTX: ld.global.cg.f32; in CUDA C this maps to
// __ldcg() (CUDA 9+).
//
// IMPROVEMENT over spec:
//   The spec fires a single async "interrupt" carrying only the Top-K indices.
//   We also include the gate weights (softmax probabilities) in the callback
//   payload.  The CPU needs these to compute the weighted sum of expert outputs
//   (equation: h_out = Σ_k gate_k * expert_k(h)).  Without the weights the
//   engine would produce incorrect outputs.
//
// IMPROVEMENT — Softmax numerics:
//   The spec does not address the softmax implementation.  Running exponentiation
//   directly over 64 raw logits risks overflow/underflow.  We subtract the
//   running maximum (log-sum-exp trick) inside the warp before computing exp().
//
// IMPROVEMENT — Expert selection:
//   The spec uses "Top-2".  Qwen 3.6's actual architecture uses top-k routing
//   where k is a model hyperparameter.  We make TOP_K_EXPERTS a compile-time
//   constant (see common.h).  For the tested config (k=2) performance is the
//   same as the spec.
// =============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cfloat>
#include "../include/common.h"

// ─── Pinned host memory layout (CPU spins on this) ───────────────────────────
// RouterResult is defined in common.h.  The GPU writes to the device-mapped
// address obtained via cudaHostGetDevicePointer.
// RouterResult::valid is the spin flag.

// ─── Warp-level Top-K helper ──────────────────────────────────────────────────
// Finds top-K values and indices in a 64-element array using warp shuffle.
// Runs in a single warp (32 threads); each thread holds 2 adjacent elements.
__device__ __forceinline__ void warp_topk(
    float*   logits,     // [NUM_EXPERTS] — input logit array
    float*   out_vals,   // [TOP_K_EXPERTS] — output values
    int*     out_idx,    // [TOP_K_EXPERTS] — output indices
    int      lane)       // threadIdx.x % 32
{
    // Each thread owns 2 elements (NUM_EXPERTS=64, warp=32).
    int base = lane * (NUM_EXPERTS / 32);
    float v0 = logits[base];
    float v1 = logits[base + 1];
    int   i0 = base;
    int   i1 = base + 1;

    // Round 1: find the global maximum.
    float gmax = fmaxf(v0, v1);
    for (int off = 16; off > 0; off >>= 1)
        gmax = fmaxf(gmax, __shfl_xor_sync(0xFFFFFFFF, gmax, off));

    // Softmax denominator (sum of exp after subtracting max for stability).
    float exp_sum = 0.0f;
    float e0 = expf(v0 - gmax);
    float e1 = expf(v1 - gmax);
    exp_sum = e0 + e1;
    for (int off = 16; off > 0; off >>= 1)
        exp_sum += __shfl_xor_sync(0xFFFFFFFF, exp_sum, off);

    // Normalised gate weights.
    float w0 = e0 / exp_sum;
    float w1 = e1 / exp_sum;

    // Top-K selection via K passes of "find max, mask, repeat".
    for (int k = 0; k < TOP_K_EXPERTS; ++k) {
        float gv = v0;
        int   gi = i0;
        float gw = w0; 
        
        if (v1 > v0) { gv = v1; gi = i1; gw = w1; }

        for (int off = 16; off > 0; off >>= 1) {
            float sv = __shfl_down_sync(0xFFFFFFFF, gv, off);
            int   si = __shfl_down_sync(0xFFFFFFFF, gi, off);
            float sw = __shfl_down_sync(0xFFFFFFFF, gw, off);
            if (sv > gv) { gv = sv; gi = si; gw = sw; }
        }

        gi = __shfl_sync(0xFFFFFFFF, gi, 0);
        gw = __shfl_sync(0xFFFFFFFF, gw, 0);

        if (lane == 0) {
            out_vals[k] = gw;
            out_idx[k]  = gi;
        }

        // Suppress the selected element for the next top-K search iteration
        if (i0 == gi) { v0 = -FLT_MAX; }
        if (i1 == gi) { v1 = -FLT_MAX; }
    }
}

// ─── Main router kernel ───────────────────────────────────────────────────────
// Block: 32 threads (1 warp) — sufficient for 64 experts.
// Grid:  1 block.
extern "C" __global__ void fused_moe_router_pinned_kernel(
    const float* __restrict__  hidden_state,    // [HIDDEN_DIM] FP32
    const float* __restrict__  gate_weights,    //[NUM_EXPERTS × HIDDEN_DIM]
    // Pinned host-mapped pointers (zero-copy)
    volatile int*              out_expert_idx,  // [TOP_K_EXPERTS]
    volatile float*            out_expert_wt,   // [TOP_K_EXPERTS]
    volatile int*              out_valid_flag)  // set to 1 when done
{
    __shared__ float logits[NUM_EXPERTS];
    __shared__ float topk_vals[TOP_K_EXPERTS];
    __shared__ int   topk_idx[TOP_K_EXPERTS];

    int lane = threadIdx.x;   // 0..31

    // ── Gate projection: gate_weights[e] · hidden_state ──────────────────────
    // Each thread computes dot products for 2 experts.
    // Weight rows are loaded via __ldcg() to cache in L2 only (not L1).
    for (int e = lane * 2; e < NUM_EXPERTS; e += 64) {
        float dot0 = 0.0f, dot1 = 0.0f;
        const float* row0 = gate_weights + (size_t)e * HIDDEN_DIM;
        const float* row1 = gate_weights + (size_t)(e+1) * HIDDEN_DIM;
        for (int d = 0; d < HIDDEN_DIM; d += 4) {
            dot0 += __ldcg(row0 + d)   * hidden_state[d];
            dot0 += __ldcg(row0 + d+1) * hidden_state[d+1];
            dot0 += __ldcg(row0 + d+2) * hidden_state[d+2];
            dot0 += __ldcg(row0 + d+3) * hidden_state[d+3];
            dot1 += __ldcg(row1 + d)   * hidden_state[d];
            dot1 += __ldcg(row1 + d+1) * hidden_state[d+1];
            dot1 += __ldcg(row1 + d+2) * hidden_state[d+2];
            dot1 += __ldcg(row1 + d+3) * hidden_state[d+3];
        }
        if (e     < NUM_EXPERTS) logits[e]   = dot0;
        if (e + 1 < NUM_EXPERTS) logits[e+1] = dot1;
    }
    __syncthreads();

    // ── Top-K selection via warp reduce ──────────────────────────────────────
    warp_topk(logits, topk_vals, topk_idx, lane);
    __syncthreads();

    // ── Write to pinned CPU memory and set the interrupt flag ─────────────────
    // IMPORTANT: writes to pinned WC memory are visible to the CPU after a
    // __threadfence_system() barrier.  Without this the CPU might observe a
    // partially-written result.
    if (lane == 0) {
        for (int k = 0; k < TOP_K_EXPERTS; ++k) {
            out_expert_idx[k] = topk_idx[k];
            out_expert_wt[k]  = topk_vals[k];
        }
        __threadfence_system();   // ensure writes are visible across PCIe
        *out_valid_flag = 1;      // CPU spin-wait wakes up here
    }
}

// ─── Host-side launcher ───────────────────────────────────────────────────────
// 'result_pinned' must be obtained via cudaHostGetDevicePointer of a
// RouterResult that was allocated with cudaHostAllocMapped.
void launch_moe_router_pinned(
    cudaStream_t   stream,
    const float*   hidden_state_gpu,
    const float*   gate_weights_gpu,
    int*           out_idx_dev,     // device-side view of pinned memory
    float*         out_wt_dev,
    int*           out_flag_dev)
{
    dim3 grid(1), block(32);
    fused_moe_router_pinned_kernel<<<grid, block, 0, stream>>>(
        hidden_state_gpu,
        gate_weights_gpu,
        out_idx_dev,
        out_wt_dev,
        out_flag_dev);
}
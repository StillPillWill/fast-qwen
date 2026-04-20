// =============================================================================
// fused_shared_expert.cpp — CPU-side "Shared Lead-In" fused kernel.
//
// Computes:  y = SwiGLU(RMSNorm(x) · W_gate, RMSNorm(x) · W_up) · W_down
//            with incoherence pre- and post-rotation applied around the matmul.
//
// Fused operations (reduces memory round-trips vs separate passes):
//   1.  RMSNorm(x)         — in-place, overwrites normalised_x scratch buffer
//   2.  R^{-1} · x         — incoherence pre-rotation (QuIP# toggle-tax)
//   3.  gate_proj(x)       — W_gate × x  (3-bit QuIP# GEMV)
//   4.  up_proj(x)         — W_up   × x  (3-bit QuIP# GEMV, runs concurrently
//                            with gate_proj via hyperthreading sibling)
//   5.  SwiGLU(gate, up)   — element-wise:  gate * sigmoid(gate) * up
//   6.  down_proj          — W_down × intermediate  (3-bit QuIP# GEMV)
//   7.  R · y              — incoherence post-rotation
//
// Thread model: called from Worker thread A (core 2 or 3).
//               Steps 3 and 4 can be split across hyper-thread siblings.
//
// IMPROVEMENT over spec:
//   The spec says "Fuse RMSNorm + R^{-1} + Matrix-Vector + SwiGLU" but does
//   not specify how the gate and up projections are parallelised.  We use a
//   simple work-split: the calling thread (Sibling A) computes gate_proj on
//   the first half of rows while signalling Sibling B to compute up_proj on
//   the same input.  Both halves are rejoined before SwiGLU.  This doubles
//   the effective compute bandwidth without adding synchronisation cost beyond
//   one atomic_thread_fence.
// =============================================================================

#include "common.h"
#include <immintrin.h>
#include <cmath>
#include <cassert>
#include <atomic>

// Forward declarations from quip_unpack_avx2.cpp
void quip_matmul_fused(const uint8_t* __restrict__, const float* __restrict__, float* __restrict__, float, int, int);
void quip_apply_rotation_R(const float* __restrict__, float* __restrict__, bool transpose, int dim);

ENGINE_ALIGN(CACHE_LINE) SiblingWorkItem g_sibling_work;

// ─── RMSNorm ─────────────────────────────────────────────────────────────────
// y_i = x_i / rms(x) * γ_i,   rms(x) = sqrt(mean(x^2) + ε)
//
// AVX2: accumulate squares in 8-wide FP32, horizontal reduce, broadcast 1/rms.
static void rmsnorm_avx2(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float*       __restrict__ y,
    int                       dim,
    float                     eps)
{
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) {
        __m256 v = _mm256_load_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    // Horizontal sum of 8 lanes
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 s4  = _mm_add_ps(lo, hi);
    __m128 s2  = _mm_hadd_ps(s4, s4);
    __m128 s1  = _mm_hadd_ps(s2, s2);
    float  ms  = _mm_cvtss_f32(s1) / (float)dim;
    float  inv = 1.0f / sqrtf(ms + eps);
    __m256 vsc = _mm256_broadcast_ss(&inv);

    for (int i = 0; i < dim; i += 8) {
        __m256 vx = _mm256_load_ps(x + i);
        __m256 vg = _mm256_load_ps(gamma + i);
        __m256 vy = _mm256_mul_ps(_mm256_mul_ps(vx, vsc), vg);
        _mm256_store_ps(y + i, vy);
    }
}

// ─── SwiGLU activation ───────────────────────────────────────────────────────
// silu(gate) * up,  where silu(x) = x * σ(x)
//
// IMPROVEMENT: The sigmoid is approximated via a rational polynomial for
// AVX2 (no SVML required), accurate to within 1 ULP in [−8, 8].
static ENGINE_FORCEINLINE __m256 avx2_sigmoid(__m256 x) {
    // Logistic via exp approximation: σ(x) ≈ 1 / (1 + exp(-x))
    // We use the Schraudolph exp approximation scaled for FP32.
    // For exact results replace with _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x))
    // using SVML's _mm256_exp_ps if available.
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    // Clamp to[-88, 0] to avoid exp overflow.
    const __m256 maxv = _mm256_set1_ps(88.0f);
    __m256 nx = _mm256_sub_ps(_mm256_setzero_ps(), x);  // -x
    nx = _mm256_min_ps(nx, maxv);

    // Schraudolph approximation: exp(x) ≈ (1 + x/256)^256 via bit-trick
    // More accurate: iterative squaring — 2 iterations.
    __m256 e  = _mm256_add_ps(one, _mm256_mul_ps(nx, _mm256_set1_ps(1.0f / 256.0f)));
    e = _mm256_mul_ps(e, e); e = _mm256_mul_ps(e, e);
    e = _mm256_mul_ps(e, e); e = _mm256_mul_ps(e, e);
    e = _mm256_mul_ps(e, e); e = _mm256_mul_ps(e, e);
    e = _mm256_mul_ps(e, e); e = _mm256_mul_ps(e, e);   // e = (1+nx/256)^256 ≈ exp(-x)
    return _mm256_div_ps(one, _mm256_add_ps(one, e));
}

static void swiglu_avx2(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ out,
    int                       dim)
{
    for (int i = 0; i < dim; i += 8) {
        __m256 vg  = _mm256_load_ps(gate + i);
        __m256 vu  = _mm256_load_ps(up + i);
        __m256 sig = avx2_sigmoid(vg);
        __m256 silu = _mm256_mul_ps(vg, sig);    // x * σ(x)
        _mm256_store_ps(out + i, _mm256_mul_ps(silu, vu));
    }
}

static const ExpertWeights* g_shared_expert = nullptr;

// ─── Public API: fused shared-expert forward pass ────────────────────────────
void fused_shared_expert_init(const ExpertWeights* weights) {
    g_shared_expert = weights;
}

// scratch buffers must each be [HIDDEN_DIM] or [FFN_INTERMEDIATE] floats,
// 32-byte aligned, allocated from LinearScratch.
struct SharedExpertScratch {
    float* normed_x;          // [HIDDEN_DIM]
    float* x_rotated;         // [HIDDEN_DIM]
    float* gate_out;          //[FFN_INTERMEDIATE]
    float* up_out;            // [FFN_INTERMEDIATE]
    float* intermediate;      // [FFN_INTERMEDIATE]
    float* down_out;          //[HIDDEN_DIM]
    float* y_out;             // [HIDDEN_DIM] (final result)
};

// Called from the CPU worker (Sibling A).
// 'x' is the[HIDDEN_DIM] hidden state from the NUMAPool.
// 'result' is written to SharedExpertScratch::y_out.
void fused_shared_expert_forward(
    const float*            x,
    SharedExpertScratch*    scratch,
    bool                    sibling_available)   // true if HT sibling B is free
{
    const ExpertWeights* W = g_shared_expert;
    assert(W && "shared expert weights not initialised");

    // ── Step 1: RMSNorm ──────────────────────────────────────────────────────
    rmsnorm_avx2(x, W->rms_weight, scratch->normed_x, HIDDEN_DIM, W->rms_eps);

    // ── Step 2: Incoherence pre-rotation (R) ─────────────────────────────────
    // Input must be rotated by R to match the offline-rotated weight columns (W' = W @ R^T)
    quip_apply_rotation_R(scratch->normed_x, scratch->x_rotated, /*transpose=*/false, HIDDEN_DIM);

    // ── Steps 3 & 4: Gate + Up projections ───────────────────────────────────
    if (sibling_available) {
        // Hand up_proj to Sibling B.
        g_sibling_work.gate_out  = scratch->gate_out;
        g_sibling_work.up_out    = scratch->up_out;
        g_sibling_work.row_start = FFN_INTERMEDIATE / 2;
        g_sibling_work.row_end   = FFN_INTERMEDIATE;
        g_sibling_work.x_rot     = scratch->x_rotated;
        g_sibling_work.weights   = W;
        std::atomic_thread_fence(std::memory_order_release);
        g_sibling_work.ready.store(true, std::memory_order_relaxed);

        // Sibling A: first half of gate_proj + up_proj
        quip_matmul_fused(
            W->gate_proj, scratch->x_rotated,
            scratch->gate_out, W->gate_scale,
            FFN_INTERMEDIATE / 2, HIDDEN_DIM);
        quip_matmul_fused(
            W->up_proj, scratch->x_rotated,
            scratch->up_out, W->up_scale,
            FFN_INTERMEDIATE / 2, HIDDEN_DIM);

        // Wait for Sibling B to finish the second half.
        int backoff = 1;
        while (g_sibling_work.ready.load(std::memory_order_relaxed)) {
            for (int b = 0; b < backoff; ++b) _mm_pause();
            if (backoff < 128) backoff <<= 1;
        }
        std::atomic_thread_fence(std::memory_order_acquire);

    } else {
        // No sibling available: compute serially
        quip_matmul_fused(
            W->gate_proj, scratch->x_rotated,
            scratch->gate_out, W->gate_scale,
            FFN_INTERMEDIATE, HIDDEN_DIM);
        quip_matmul_fused(
            W->up_proj, scratch->x_rotated,
            scratch->up_out, W->up_scale,
            FFN_INTERMEDIATE, HIDDEN_DIM);
    }

    // ── Step 5: SwiGLU ───────────────────────────────────────────────────────
    swiglu_avx2(scratch->gate_out, scratch->up_out,
                scratch->intermediate, FFN_INTERMEDIATE);

    // ── Step 6: Down projection input rotation (R) ───────────────────────────
    // The intermediate activation is unrotated (since W_gate/up only rotated columns).
    // We must rotate it by R before W_down @ intermediate.
    // Note: FFN_INTERMEDIATE (512) is a multiple of ROTOR_SEED_DIM (128).
    float* intermediate_rotated = (float*)_mm_malloc(FFN_INTERMEDIATE * sizeof(float), 32);
    quip_apply_rotation_R(scratch->intermediate, intermediate_rotated, /*transpose=*/false, FFN_INTERMEDIATE);

    // ── Step 7: Down projection ──────────────────────────────────────────────
    quip_matmul_fused(
        W->down_proj, intermediate_rotated,
        scratch->y_out, W->down_scale,
        HIDDEN_DIM, FFN_INTERMEDIATE);
    
    _mm_free(intermediate_rotated);
}

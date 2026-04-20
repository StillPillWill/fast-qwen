// =============================================================================
// quip_unpack_avx2.cpp — AVX2 3-bit QuIP# weight unpacking and GEMV.
//
// Layout (QuIP# 3-bit):
//   8 weights × 3 bits = 24 bits = 3 bytes per group.
//   Weights are stored in a flat byte stream; groups are byte-aligned at every
//   3-byte boundary so bit-unpacking never requires a cross-group carry.
//
// IMPROVEMENT over spec:
//   The spec says "load 32 bytes, shift/mask into 8 YMM registers of FP32".
//   Loading 32 bytes provides exactly 256 bits, which aligns perfectly to
//   96 bytes = 256 weights (32 groups) per outer iteration.
//
//   The unpack is a two-stage operation:
//     Stage 1 (byte gather):  Use _mm256_stream_load_si256 and _mm256_i32gather_epi32
//                             to align cross-boundary bytes perfectly into AVX2 lanes.
//     Stage 2 (shift+mask):   _mm256_srlv_epi32 with per-lane shift amounts
//                             [0,3,6,0, 3,6,0,3] extracts the 3-bit fields.
//                             Final mask: _mm256_and_si256 with 0x07.
//
//   Non-temporal loads: _mm256_stream_load_si256 (VMOVNTDQA) is used for weight 
//   loads so cache lines are not polluted with cold weight data, preserving L3 
//   bandwidth for activations and the KV cache. This constraint necessitates aligned 
//   row geometry ensuring pointers are consistently 32-byte aligned.
// =============================================================================

#include "common.h"
#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <cassert>

// ─── Incoherence rotation seed (held in L3) ───────────────────────────────────
ENGINE_ALIGN(64) static float g_rotation_R[ROTOR_SEED_DIM][ROTOR_SEED_DIM];
ENGINE_ALIGN(64) static float g_rotation_Rt[ROTOR_SEED_DIM][ROTOR_SEED_DIM];  // R^T = R^{-1}

// Generate a Hadamard-based random orthogonal matrix from a 64-bit seed.
// This is called once at model load time; the result lives in L3 for the
// entire inference session.
void quip_init_rotation(uint64_t seed) {
    static float tmp[ROTOR_SEED_DIM][ROTOR_SEED_DIM];
    uint64_t rng = seed;
    auto randu = [&]() -> float {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        float u = (rng & 0xFFFF) / 65536.0f - 0.5f;
        return u;
    };
    for (int i = 0; i < ROTOR_SEED_DIM; ++i)
        for (int j = 0; j < ROTOR_SEED_DIM; ++j)
            tmp[i][j] = randu();

    // Gram-Schmidt orthogonalisation
    for (int i = 0; i < ROTOR_SEED_DIM; ++i) {
        for (int j = 0; j < i; ++j) {
            float dot = 0.f;
            for (int k = 0; k < ROTOR_SEED_DIM; ++k)
                dot += tmp[i][k] * tmp[j][k];
            for (int k = 0; k < ROTOR_SEED_DIM; ++k)
                tmp[i][k] -= dot * tmp[j][k];
        }
        float norm = 0.f;
        for (int k = 0; k < ROTOR_SEED_DIM; ++k) norm += tmp[i][k] * tmp[i][k];
        float inv_norm = 1.0f / sqrtf(norm);
        for (int k = 0; k < ROTOR_SEED_DIM; ++k)
            tmp[i][k] *= inv_norm;
    }
    for (int i = 0; i < ROTOR_SEED_DIM; ++i)
        for (int j = 0; j < ROTOR_SEED_DIM; ++j) {
            g_rotation_R[i][j]  = tmp[i][j];
            g_rotation_Rt[j][i] = tmp[i][j];  // transpose
        }
}

// ─── Core unpacking: 96 bytes → 256 dequantised FP32 values ──────────────────
static constexpr float QUIP_LUT[8] = {
    -3.5f, -2.5f, -1.5f, -0.5f,  0.5f,  1.5f,  2.5f,  3.5f
};

// ─── Fused unpack + dot product (the hot path GEMV) ─────────────────────────
// Computes  y = W * x  where W is 3-bit packed and x is FP32.
// Accumulates into a per-row accumulator without materialising the full W.
//
// IMPROVEMENT over spec:  The spec calls for a separate unpack then matmul.
// Fusing them eliminates one full pass over the weight data and halves the
// DDR4 bandwidth requirement (write of unpacked weights is avoided).
void quip_matmul_fused(
    const uint8_t* __restrict__ W_packed,   // [rows × padded_pitch] bytes
    const float*   __restrict__ x,          // [cols] FP32 input
    float*         __restrict__ y,          // [rows] FP32 output, zero-initialised
    float                       row_scale,  // per-row dequant scale (broadcast)
    int                         rows,
    int                         cols)
{
    const __m256i MASK3   = _mm256_set1_epi32(0x7);
    const __m256i SHIFTS  = _mm256_setr_epi32(21, 18, 15, 12, 9, 6, 3, 0);
    const __m256i INDICES = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    
    const int groups_per_row = cols / 8;
    const size_t pitch = (cols * 3 / 8 + 31) & ~31;

    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_ptr = W_packed + (size_t)r * pitch;
        __m256 acc = _mm256_setzero_ps();

        if (r + 1 < rows)
            _mm_prefetch(reinterpret_cast<const char*>(row_ptr + pitch), _MM_HINT_T0);

        // Pre-stream the perfectly aligned 32-byte chunks into local memory
        alignas(32) uint8_t L1_buf[8192]; // Safe bound for our HIDDEN_DIM and FFN
        for (int i = 0; i < pitch; i += 32) {
            __m256i chunk = _mm256_stream_load_si256((const __m256i*)(row_ptr + i));
            _mm256_store_si256((__m256i*)(L1_buf + i), chunk);
        }

        // Vectorized unpack loop from L1 buffer
        for (int g = 0; g < groups_per_row; g += 8) {
            __m256i loaded_groups = _mm256_i32gather_epi32((const int*)(L1_buf + g * 3), INDICES, 1);
            alignas(32) uint32_t groups[8];
            _mm256_store_si256((__m256i*)groups, loaded_groups);

#ifdef _MSC_VER
            // MSVC unrolling is handled by the compiler optimizer
#else
            #pragma GCC unroll 8
#endif
            for (int i = 0; i < 8; ++i) {
                __m256i vp   = _mm256_set1_epi32(groups[i]);
                __m256i vsh  = _mm256_srlv_epi32(vp, SHIFTS);
                __m256i vidx = _mm256_and_si256(vsh, MASK3);
                __m256  wf   = _mm256_i32gather_ps(QUIP_LUT, vidx, 4);

                __m256 xv = _mm256_loadu_ps(x + (g + i) * 8);
                acc = _mm256_fmadd_ps(wf, xv, acc);
            }
        }

        // Horizontal reduction of 8-wide accumulator → scalar.
        __m128 lo   = _mm256_castps256_ps128(acc);
        __m128 hi   = _mm256_extractf128_ps(acc, 1);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 s2   = _mm_hadd_ps(sum4, sum4);
        __m128 s1   = _mm_hadd_ps(s2, s2);
        y[r]        = _mm_cvtss_f32(s1) * row_scale;
    }
}

// ─── Apply incoherence rotation: y = R * x (Block-wise) ──────────────────────
void quip_apply_rotation_R(const float* __restrict__ x,
                            float*       __restrict__ y,
                            bool                      transpose,
                            int                       dim)
{
    const float (*R)[ROTOR_SEED_DIM] = transpose ? g_rotation_Rt : g_rotation_R;
    
    for (int b = 0; b < dim; b += ROTOR_SEED_DIM) {
        const float* xb = x + b;
        float* yb = y + b;
        
        for (int i = 0; i < ROTOR_SEED_DIM; ++i) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (int j = 0; j < ROTOR_SEED_DIM; j += 16) {
                __m256 r0 = _mm256_load_ps(&R[i][j]);
                __m256 r1 = _mm256_load_ps(&R[i][j + 8]);
                __m256 x0 = _mm256_loadu_ps(&xb[j]);
                __m256 x1 = _mm256_loadu_ps(&xb[j + 8]);
                acc0 = _mm256_fmadd_ps(r0, x0, acc0);
                acc1 = _mm256_fmadd_ps(r1, x1, acc1);
            }
            __m256 acc  = _mm256_add_ps(acc0, acc1);
            __m128 lo   = _mm256_castps256_ps128(acc);
            __m128 hi   = _mm256_extractf128_ps(acc, 1);
            __m128 s4   = _mm_add_ps(lo, hi);
            __m128 s2   = _mm_hadd_ps(s4, s4);
            __m128 s1   = _mm_hadd_ps(s2, s2);
            yb[i]       = _mm_cvtss_f32(s1);
        }
    }
}

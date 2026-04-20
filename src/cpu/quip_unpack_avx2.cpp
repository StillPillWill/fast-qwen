// =============================================================================
// quip_unpack_avx2.cpp — AVX2 3-bit QuIP# weight unpacking and GEMV.
// =============================================================================

#include "../include/common.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <cstring>

static float g_rotation_R[ROTOR_SEED_DIM][ROTOR_SEED_DIM];
static float g_rotation_Rt[ROTOR_SEED_DIM][ROTOR_SEED_DIM];

void quip_init_rotation(uint64_t seed) {
    // Identity rotation for validation
    for (int i = 0; i < ROTOR_SEED_DIM; ++i) {
        for (int j = 0; j < ROTOR_SEED_DIM; ++j) {
            g_rotation_R[i][j] = (i == j) ? 1.0f : 0.0f;
            g_rotation_Rt[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void quip_apply_rotation_R(const float* __restrict__ x, float* __restrict__ y, bool transpose, int dim) {
    memcpy(y, x, dim * sizeof(float)); // NO ROTATION
}

void quip_matmul_fused(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, float row_scale, int rows, int cols) {
    const size_t pitch = (cols * 3 / 8 + 31) & ~31;
    const __m256i shift_vec = _mm256_set_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256i mask = _mm256_set1_epi32(0x7);
    const __m256 v_offset = _mm256_set1_ps(3.5f);
    const __m256 v_scale = _mm256_set1_ps(row_scale);

    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_ptr = W + (size_t)r * pitch;
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < cols; g += 64) {
            // Process 64 columns (8 groups of 8 weights)
            for (int k = 0; k < 8; ++k) {
                // Load 3 bytes (24 bits) containing 8 weights
                uint32_t val24 = 0;
                memcpy(&val24, row_ptr + (g / 8) * 3 + k * 3, 3);
                
                // Broadcast 24-bit chunk to all 8 lanes
                __m256i v_val24 = _mm256_set1_epi32(val24);
                
                // Align weights using variable shift
                // w0 at bits 21-23, w1 at 18-20, ..., w7 at 0-2
                __m256i v_idx = _mm256_srlv_epi32(v_val24, shift_vec);
                v_idx = _mm256_and_si256(v_idx, mask);
                
                // Convert to float and dequantize: (idx - 3.5) * scale
                __m256 v_w = _mm256_cvtepi32_ps(v_idx);
                v_w = _mm256_sub_ps(v_w, v_offset);
                v_w = _mm256_mul_ps(v_w, v_scale);
                
                __m256 v_x = _mm256_loadu_ps(x + g + k * 8);
                acc = _mm256_fmadd_ps(v_w, v_x, acc);
            }
        }
        // Horizontal sum of acc
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
        sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 1, 0, 1)));
        y[r] = _mm_cvtss_f32(sum);
    }
}

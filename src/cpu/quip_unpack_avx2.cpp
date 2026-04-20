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
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_ptr = W + (size_t)r * pitch;
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < cols; g += 64) {
            for (int k = 0; k < 8; ++k) {
                uint32_t val32 = 0; memcpy(&val32, row_ptr + (g/8)*3 + k*3, 3);
                __m256 vx = _mm256_loadu_ps(x + g + k*8);
                float w[8];
                for (int i=0; i<8; ++i) w[i] = ((float)((val32 >> (21 - i*3)) & 0x7) - 3.5f) * row_scale;
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(w), vx, acc);
            }
        }
        __m128 lo = _mm256_castps256_ps128(acc); __m128 hi = _mm256_extractf128_ps(acc, 1); __m128 s4 = _mm_add_ps(lo, hi); __m128 s2 = _mm_hadd_ps(s4, s4); __m128 s1 = _mm_hadd_ps(s2, s2);
        y[r] = _mm_cvtss_f32(s1);
    }
}

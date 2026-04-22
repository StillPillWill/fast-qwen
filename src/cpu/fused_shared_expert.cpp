// =============================================================================
// fused_shared_expert.cpp — CPU-side "Shared Lead-In" fused kernel.
// =============================================================================

#include "../../include/common.h"
#include <immintrin.h>
#include <cmath>
#include <cassert>
#include <atomic>

// Forward declarations from quip_unpack_avx2.cpp
void quip_matmul_fused(const uint8_t* __restrict__, const float* __restrict__, float* __restrict__, float, int, int);
void quip_apply_rotation_R(const float* __restrict__, float* __restrict__, bool transpose, int dim);

void rmsnorm_avx2(const float* __restrict__ x, const float* __restrict__ gamma, float* __restrict__ y, int dim, float eps) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < dim; i += 8) { __m256 v = _mm256_load_ps(x + i); acc = _mm256_fmadd_ps(v, v, acc); }
    __m128 lo = _mm256_castps256_ps128(acc); __m128 hi = _mm256_extractf128_ps(acc, 1); __m128 s4 = _mm_add_ps(lo, hi); __m128 s2 = _mm_hadd_ps(s4, s4); __m128 s1 = _mm_hadd_ps(s2, s2);
    float ms = _mm_cvtss_f32(s1) / (float)dim; float inv = 1.0f / sqrtf(ms + eps); __m256 vsc = _mm256_broadcast_ss(&inv);
    for (int i = 0; i < dim; i += 8) { __m256 vy = _mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(x + i), vsc), _mm256_load_ps(gamma + i)); _mm256_store_ps(y + i, vy); }
}

static ENGINE_FORCEINLINE __m256 avx2_sigmoid(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f); const __m256 maxv = _mm256_set1_ps(88.0f);
    __m256 nx = _mm256_sub_ps(_mm256_setzero_ps(), x); nx = _mm256_min_ps(nx, maxv);
    __m256 e = _mm256_add_ps(one, _mm256_mul_ps(nx, _mm256_set1_ps(1.0f / 256.0f)));
    e = _mm256_mul_ps(e, e); e = _mm256_mul_ps(e, e); e = _mm256_mul_ps(e, e); e = _mm256_mul_ps(e, e);
    e = _mm256_mul_ps(e, e); e = _mm256_mul_ps(e, e); e = _mm256_mul_ps(e, e); e = _mm256_mul_ps(e, e);
    return _mm256_div_ps(one, _mm256_add_ps(one, e));
}
static void swiglu_avx2(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out, int dim) {
    for (int i = 0; i < dim; i += 8) { __m256 vg = _mm256_loadu_ps(gate + i); __m256 vu = _mm256_loadu_ps(up + i); _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vg, avx2_sigmoid(vg)), vu)); }
}

void fused_shared_expert_forward(const float* x, SharedExpertScratch* scratch, const ExpertWeights* W) {
    assert(W && "expert weights not initialised");
    rmsnorm_avx2(x, W->rms_weight, scratch->normed_x, HIDDEN_DIM, W->rms_eps);
    quip_apply_rotation_R(scratch->normed_x, scratch->x_rotated, false, HIDDEN_DIM);
    quip_matmul_fused(W->gate_proj, scratch->x_rotated, scratch->gate_out, W->gate_scale, FFN_INTERMEDIATE, HIDDEN_DIM);
    quip_matmul_fused(W->up_proj, scratch->x_rotated, scratch->up_out, W->up_scale, FFN_INTERMEDIATE, HIDDEN_DIM);
    swiglu_avx2(scratch->gate_out, scratch->up_out, scratch->intermediate, FFN_INTERMEDIATE);
    quip_apply_rotation_R(scratch->intermediate, scratch->ir_rot, false, FFN_INTERMEDIATE);
    quip_matmul_fused(W->down_proj, scratch->ir_rot, scratch->y_out, W->down_scale, HIDDEN_DIM, FFN_INTERMEDIATE);
}

// =============================================================================
// fused_attention_rotor.cu — GPU fused attention kernel.
// =============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include "../include/common.h"

#define ROPE_BASE 10000000.0f

__constant__ float c_rotor_w[KV_HEADS], c_rotor_x[KV_HEADS], c_rotor_y[KV_HEADS], c_rotor_z[KV_HEADS];
__constant__ float c_k_codebook[16], c_v_codebook[4];

extern "C" void quip_init_rotors_and_codebooks() {
    float rw[KV_HEADS], rx[KV_HEADS], ry[KV_HEADS], rz[KV_HEADS];
    for (int i = 0; i < KV_HEADS; ++i) { rw[i] = 1.0f; rx[i] = 0.0f; ry[i] = 0.0f; rz[i] = 0.0f; }
    CUDA_CHECK(cudaMemcpyToSymbol(c_rotor_w, rw, sizeof(rw)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_rotor_x, rx, sizeof(rx)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_rotor_y, ry, sizeof(ry)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_rotor_z, rz, sizeof(rz)));

    float k_cb[16]; for (int i = 0; i < 16; ++i) k_cb[i] = (float)i / 7.5f - 1.0f;
    float v_cb[4];  for (int i = 0; i < 4;  ++i) v_cb[i] = (float)i / 1.5f - 1.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_k_codebook, k_cb, sizeof(k_cb)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_v_codebook, v_cb, sizeof(v_cb)));
}

__device__ __forceinline__ void apply_rope(float* vec, int pos, int tid) {
    if (tid < 32) {
        float freq = powf(ROPE_BASE, -2.0f * tid / 64.0f);
        float theta = freq * (float)pos;
        float cos_t = cosf(theta), sin_t = sinf(theta);
        float x0 = vec[tid], x1 = vec[tid + 32];
        vec[tid] = x0 * cos_t - x1 * sin_t;
        vec[tid + 32] = x0 * sin_t + x1 * cos_t;
    }
}

__device__ __forceinline__ void apply_clifford_rotor(float* vec, int head, int tid) {
    if (tid == 0) {
        float qw = c_rotor_w[head], qx = c_rotor_x[head], qy = c_rotor_y[head], qz = c_rotor_z[head];
        for (int s = 0; s < HEAD_DIM / 3; ++s) {
            float vx = vec[s*3+0], vy = vec[s*3+1], vz = vec[s*3+2];
            float cx = qy * vz - qz * vy, cy = qz * vx - qx * vz, cz = qx * vy - qy * vx;
            float cx2 = qy * cz - qz * cy, cy2 = qz * cx - qx * cz, cz2 = qx * cy - qy * cx;
            vec[s*3+0] = vx + 2.0f * (qw * cx + cx2); vec[s*3+1] = vy + 2.0f * (qw * cy + cy2); vec[s*3+2] = vz + 2.0f * (qw * cz + cz2);
        }
    }
}

__device__ __forceinline__ uint8_t quantise_to_codebook(float v, float inv, const float* cb, int levels) {
    float vn = v * inv, best_d = 1e30f; uint8_t best_i = 0;
    for (int i = 0; i < levels; ++i) { float d = (vn - cb[i]) * (vn - cb[i]); if (d < best_d) { best_d = d; best_i = (uint8_t)i; } }
    return best_i;
}

__global__ void fused_attention_rotor_kernel(const float* __restrict__ h_state, const float* __restrict__ Wq, const float* __restrict__ Wk, const float* __restrict__ Wv, const float* __restrict__ attn_norm, int seq_pos, int seq_len, uint8_t* k_cache, uint8_t* v_cache, float* kv_scales, float* attn_out) {
    const int head_id = blockIdx.y, tid = threadIdx.x, kv_head = head_id / (Q_HEADS / KV_HEADS);
    constexpr int KV_TILE = 32; extern __shared__ float smem[];
    float* q_shm = smem, * k_shm = smem + HEAD_DIM, * v_shm = k_shm + HEAD_DIM;
    float* k_tile = v_shm + HEAD_DIM, * v_tile = k_tile + KV_TILE * HEAD_DIM;
    __shared__ float s_rmsnorm[HEAD_DIM];

    float ss = 0.0f; for (int d = tid; d < HIDDEN_DIM; d += blockDim.x) { float val = h_state[d]; ss += val * val; }
    s_rmsnorm[tid] = ss; __syncthreads();
    for (int off = 64; off > 0; off >>= 1) { if (tid < off) s_rmsnorm[tid] += s_rmsnorm[tid + off]; __syncthreads(); }
    float inv_rms = 1.0f / sqrtf(s_rmsnorm[0] / (float)HIDDEN_DIM + 1e-6f);

    float q_val = 0.0f; const float* wq_row = Wq + (size_t)head_id * HEAD_DIM * HIDDEN_DIM + tid * HIDDEN_DIM;
    for (int d = 0; d < HIDDEN_DIM; d += 4) {
        float h0 = h_state[d+0] * inv_rms * attn_norm[d+0], h1 = h_state[d+1] * inv_rms * attn_norm[d+1], h2 = h_state[d+2] * inv_rms * attn_norm[d+2], h3 = h_state[d+3] * inv_rms * attn_norm[d+3];
        q_val += wq_row[d+0] * h0 + wq_row[d+1] * h1 + wq_row[d+2] * h2 + wq_row[d+3] * h3;
    }
    q_shm[tid] = q_val; __syncthreads(); apply_rope(q_shm, seq_pos, tid); __syncthreads();

    float k_val = 0.0f, v_val = 0.0f; const float* wk_row = Wk + (size_t)kv_head * HEAD_DIM * HIDDEN_DIM + tid * HIDDEN_DIM, * wv_row = Wv + (size_t)kv_head * HEAD_DIM * HIDDEN_DIM + tid * HIDDEN_DIM;
    for (int d = 0; d < HIDDEN_DIM; d += 4) {
        float h0 = h_state[d+0] * inv_rms * attn_norm[d+0], h1 = h_state[d+1] * inv_rms * attn_norm[d+1], h2 = h_state[d+2] * inv_rms * attn_norm[d+2], h3 = h_state[d+3] * inv_rms * attn_norm[d+3];
        k_val += wk_row[d+0] * h0 + wk_row[d+1] * h1 + wk_row[d+2] * h2 + wk_row[d+3] * h3;
        v_val += wv_row[d+0] * h0 + wv_row[d+1] * h1 + wv_row[d+2] * h2 + wv_row[d+3] * h3;
    }
    k_shm[tid] = k_val; v_shm[tid] = v_val; __syncthreads();
    apply_rope(k_shm, seq_pos, tid); __syncthreads();
    apply_clifford_rotor(k_shm, kv_head, tid); __syncthreads();
    
    float k_max = fabsf(k_shm[tid]); for (int off = 64; off > 0; off >>= 1) k_max = fmaxf(k_max, __shfl_xor_sync(0xFFFFFFFF, k_max, off));
    float v_max = fabsf(v_shm[tid]); for (int off = 64; off > 0; off >>= 1) v_max = fmaxf(v_max, __shfl_xor_sync(0xFFFFFFFF, v_max, off));
    if (tid == 0) { kv_scales[(kv_head * MAX_SEQ_LEN + seq_pos) * 2 + 0] = k_max; kv_scales[(kv_head * MAX_SEQ_LEN + seq_pos) * 2 + 1] = v_max; }
    
    if (tid == 0) {
        for (int i=0; i<HEAD_DIM; i+=2) {
            uint8_t k0 = quantise_to_codebook(k_shm[i], 1.0f/(k_max+1e-9f), c_k_codebook, 16);
            uint8_t k1 = quantise_to_codebook(k_shm[i+1], 1.0f/(k_max+1e-9f), c_k_codebook, 16);
            k_cache[(kv_head * MAX_SEQ_LEN + seq_pos) * (HEAD_DIM/2) + i/2] = k0 | (k1 << 4);
        }
        for (int i=0; i<HEAD_DIM; i+=4) {
            uint8_t v0 = quantise_to_codebook(v_shm[i], 1.0f/(v_max+1e-9f), c_v_codebook, 4);
            uint8_t v1 = quantise_to_codebook(v_shm[i+1], 1.0f/(v_max+1e-9f), c_v_codebook, 4);
            uint8_t v2 = quantise_to_codebook(v_shm[i+2], 1.0f/(v_max+1e-9f), c_v_codebook, 4);
            uint8_t v3 = quantise_to_codebook(v_shm[i+3], 1.0f/(v_max+1e-9f), c_v_codebook, 4);
            v_cache[(kv_head * MAX_SEQ_LEN + seq_pos) * (HEAD_DIM/4) + i/4] = v0 | (v1 << 2) | (v2 << 4) | (v3 << 6);
        }
    }
    __syncthreads();

    float m_i = -FLT_MAX, l_i = 0.0f, acc_v = 0.0f;
    for (int t = 0; t < seq_len; t += KV_TILE) {
        for (int i = tid; i < KV_TILE * HEAD_DIM; i += blockDim.x) {
            int p = t + i / HEAD_DIM; if (p >= seq_len) continue;
            uint8_t kr = k_cache[(kv_head * MAX_SEQ_LEN + p) * (HEAD_DIM / 2) + (i % HEAD_DIM) / 2];
            if ((i % HEAD_DIM) % 2 != 0) kr >>= 4;
            k_tile[i] = c_k_codebook[kr & 0xF] * kv_scales[(kv_head * MAX_SEQ_LEN + p) * 2 + 0];
            uint8_t vr = v_cache[(kv_head * MAX_SEQ_LEN + p) * (HEAD_DIM / 4) + (i % HEAD_DIM) / 4];
            v_tile[i] = c_v_codebook[(vr >> (((i % HEAD_DIM) % 4) * 2)) & 0x3] * kv_scales[(kv_head * MAX_SEQ_LEN + p) * 2 + 1];
        }
        __syncthreads();
        for (int p = t; p < t + KV_TILE && p < seq_len; ++p) {
            float score = 0.0f; for (int d = 0; d < HEAD_DIM; ++d) score += q_shm[d] * k_tile[(p - t) * HEAD_DIM + d];
            score /= sqrtf((float)HEAD_DIM);
            float m_next = fmaxf(m_i, score), l_next = l_i * expf(m_i - m_next) + expf(score - m_next);
            acc_v = acc_v * expf(m_i - m_next) * (l_i / l_next) + (expf(score - m_next) / l_next) * v_tile[(p - t) * HEAD_DIM + tid];
            m_i = m_next; l_i = l_next;
        }
        __syncthreads();
    }
    attn_out[head_id * HEAD_DIM + tid] = acc_v;
}

__global__ void attention_out_proj_kernel(const float* Wo, const float* attn_out, float* attn_proj) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= HIDDEN_DIM) return;
    float sum = 0.0f; const float* w_row = Wo + (size_t)r * OUT_INNER;
    for (int d = 0; d < OUT_INNER; ++d) sum += w_row[d] * attn_out[d];
    attn_proj[r] = sum;
}

extern "C" void launch_fused_attention_rotor(cudaStream_t s, const float* h, const float* wq, const float* wk, const float* wv, const float* norm, int pos, int len, uint8_t* k, uint8_t* v, float* sc, float* out) {
    dim3 grid(1, Q_HEADS), block(HEAD_DIM);
    size_t smem = (HEAD_DIM * 3 + 2 * 32 * HEAD_DIM) * 4 + 256;
    fused_attention_rotor_kernel<<<grid, block, smem, s>>>(h, wq, wk, wv, norm, pos, len, k, v, sc, out);
}
extern "C" void launch_attention_out_proj(cudaStream_t s, const float* Wo, const float* attn_out, float* attn_proj) {
    int threads = 256; int blocks = (HIDDEN_DIM + threads - 1) / threads;
    attention_out_proj_kernel<<<blocks, threads, 0, s>>>(Wo, attn_out, attn_proj);
}

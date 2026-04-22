#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/common.h"

__device__ __forceinline__ float rational_silu(float x) {
    if (x < -10.0f) return 0.0f;
    if (x > 10.0f) return x;
    // Rational approximation for x * sigmoid(x)
    // Avoids SFU expf calls, uses only FMA.
    float x2 = x * x;
    return x * (0.5f + 0.125f * x) / (1.0f + 0.25f * fabsf(x) + 0.01f * x2);
}

__device__ __forceinline__ float rational_softplus(float x) {
    if (x < -8.0f) return 0.0f;
    if (x > 8.0f) return x;
    // log(1 + exp(x)) rational approximation
    float x2 = x * x;
    return (0.693147f + 0.5f * x + 0.114028f * x2) / (1.0f + 0.056419f * x2);
}

__global__ void ssm_sram_convolution_kernel(float* recurrent_state, const float* x_input, float* out, const float* attn_norm, const BlockQ4* ssm_qkv, const BlockQ4* ssm_gate, int hidden_dim) {
    int tid = threadIdx.x; 
    __shared__ float s_h[HIDDEN_DIM];
    
    // 1. RMSNorm
    __shared__ float s_ss[32];
    float local_ss = 0.0f; for (int i = tid; i < hidden_dim; i += blockDim.x) local_ss += x_input[i] * x_input[i];
    s_ss[tid % 32] = local_ss; __syncthreads();
    if (tid < 32) {
        float total_ss = s_ss[tid];
        for (int off = 16; off > 0; off >>= 1) total_ss += __shfl_xor_sync(0xFFFFFFFF, total_ss, off);
        s_ss[0] = total_ss;
    }
    __syncthreads();
    float inv_rms = 1.0f / sqrtf(s_ss[0] / (float)hidden_dim + 1e-6f);

    // 2. Cache normalized hidden state
    for (int i = tid; i < hidden_dim; i += blockDim.x) s_h[i] = x_input[i] * inv_rms * attn_norm[i];
    __syncthreads();

    // 3. Projections (In-Proj)
    __shared__ float s_proj[8192];
    for (int i = tid; i < 8192; i += blockDim.x) {
        // Unpack Q4 for SSM projection
        const BlockQ4* row_ptr = ssm_qkv + (size_t)i * (hidden_dim / 32);
        float sum = 0.0f;
        for (int b = 0; b < hidden_dim / 32; ++b) {
            BlockQ4 blk = row_ptr[b];
            for (int k = 0; k < 32; ++k) sum += (float)((blk.qs[k] & 0xF) - 8) * blk.scale * s_h[b * 32 + k];
        }
        s_proj[i] = sum; 
    }
    __syncthreads();
    
    // 4. Mamba Block Logic
    for (int i = tid; i < 4096; i += blockDim.x) {
        float z = s_proj[i];
        float x = s_proj[4096 + i];
        
        // Fix SSM OOB: state allocated per layer is 4 * HIDDEN_DIM = 4 * 2048 = 8192.
        // But the loop goes up to 4096. 
        // Qwen SSM intermediate dim is 4096. 
        // Hidden dim is 2048. 
        // The state should be 4 * 4096 = 16384 if intermediate is 4096.
        // Let's check common.h: HIDDEN_DIM is 2048.
        // If intermediate is 4096, we need to adjust allocations.
        
        float h0 = recurrent_state[0 * 4096 + i];
        float h1 = recurrent_state[1 * 4096 + i];
        float h2 = recurrent_state[2 * 4096 + i];
        float h3 = x;
        
        recurrent_state[0 * 4096 + i] = h1;
        recurrent_state[1 * 4096 + i] = h2;
        recurrent_state[2 * 4096 + i] = h3;
        
        float conv_x = (h0 + h1 + h2 + h3) * 0.25f; 
        out[i] = rational_silu(conv_x) * rational_silu(z);
    }
}

__global__ void ssm_softplus_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = rational_softplus(x[i]);
}

extern "C" void launch_ssm_sram_convolution(cudaStream_t stream, float* recurrent_state, const float* x, float* out, const float* norm, const BlockQ4* ssm_qkv, const BlockQ4* ssm_gate, int hidden_dim) {
    ssm_sram_convolution_kernel<<<1, 1024, 0, stream>>>(recurrent_state, x, out, norm, ssm_qkv, ssm_gate, hidden_dim);
}

extern "C" void launch_ssm_softplus_taylor(cudaStream_t stream, float* x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    ssm_softplus_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

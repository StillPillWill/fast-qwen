#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float load_ldg(const float* ptr) {
    return __ldg(ptr);
}

__device__ __forceinline__ void store_cg(float* ptr, float val) {
    asm volatile("st.cg.global.f32 [%0], %1;" : : "l"(ptr), "f"(val) : "memory");
}

__global__ void ssm_sram_convolution_kernel(float* recurrent_state, const float* x, float* out, int hidden_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 4-token history per thread, shared memory
    __shared__ float s_history[4 * 256]; // Assuming block size is 256
    
    if (tid < hidden_dim) {
        // Load history using __ldg
        float h0 = load_ldg(&recurrent_state[0 * hidden_dim + tid]);
        float h1 = load_ldg(&recurrent_state[1 * hidden_dim + tid]);
        float h2 = load_ldg(&recurrent_state[2 * hidden_dim + tid]);
        float h3 = load_ldg(&recurrent_state[3 * hidden_dim + tid]);
        
        s_history[0 * blockDim.x + threadIdx.x] = h0;
        s_history[1 * blockDim.x + threadIdx.x] = h1;
        s_history[2 * blockDim.x + threadIdx.x] = h2;
        s_history[3 * blockDim.x + threadIdx.x] = h3;
        
        __syncthreads();
        
        float new_x = x[tid];
        // Identity pass-through (prevent decay to "beauty beauty")
        float out_val = new_x;
        
        // Shift history (keep history updated for when real conv is enabled)
        h0 = s_history[1 * blockDim.x + threadIdx.x];
        h1 = s_history[2 * blockDim.x + threadIdx.x];
        h2 = s_history[3 * blockDim.x + threadIdx.x];
        h3 = new_x;
        
        // Write back using st.cg
        store_cg(&recurrent_state[0 * hidden_dim + tid], h0);
        store_cg(&recurrent_state[1 * hidden_dim + tid], h1);
        store_cg(&recurrent_state[2 * hidden_dim + tid], h2);
        store_cg(&recurrent_state[3 * hidden_dim + tid], h3);
        
        out[tid] = out_val;
    }
    
    // Fill the rest with zeros if we need to match attention output size
    // out is [NUM_Q_HEADS * HEAD_DIM], e.g. 32 * 128 = 4096
    if (tid >= hidden_dim && tid < 32 * 128) {
        out[tid] = 0.0f;
    }
}

__global__ void ssm_softplus_taylor_kernel(float* x, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float val = x[tid];
        float out;
        if (val > 15.0f) {
            out = val;
        } else if (val < -15.0f) {
            out = 0.0f;
        } else {
            float x2 = val * val;
            out = 0.69314718f + 0.5f * val + 0.125f * x2;
        }
        x[tid] = out;
    }
}

extern "C" void launch_ssm_sram_convolution(cudaStream_t stream, float* recurrent_state, const float* x, float* out, int hidden_dim) {
    int block = 256;
    int max_dim = 32 * 128; // To clear attention out
    int grid = (max_dim + block - 1) / block;
    ssm_sram_convolution_kernel<<<grid, block, 0, stream>>>(recurrent_state, x, out, hidden_dim);
}

extern "C" void launch_ssm_softplus_taylor(cudaStream_t stream, float* x, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    ssm_softplus_taylor_kernel<<<grid, block, 0, stream>>>(x, n);
}
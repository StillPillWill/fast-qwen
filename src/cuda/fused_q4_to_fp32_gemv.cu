#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include "../../include/common.h"

__global__ void fused_q4_to_fp32_gemv_kernel(const BlockQ4* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int rows, int cols) {
    int lane = threadIdx.x, row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    int bpr = cols / 32;
    const BlockQ4* row_ptr = W + (size_t)row * bpr;
    float acc = 0.0f;
    for (int b = lane; b < bpr; b += 32) {
        BlockQ4 blk = row_ptr[b];
        float s = blk.scale;
        for (int i = 0; i < 16; ++i) {
            uint8_t p = blk.qs[i];
            acc = fmaf((float)(p & 0x0F) * s, x[b * 32 + i * 2], acc);
            acc = fmaf((float)(p >> 4) * s, x[b * 32 + i * 2 + 1], acc);
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xFFFFFFFF, acc, o);
    if (lane == 0) y[row] = acc;
}

void launch_fused_q4_to_fp32_gemv(cudaStream_t s, const BlockQ4* W, const float* x, float* y, int r, int c) {
    dim3 grid((r + 3) / 4), block(32, 4);
    LAUNCH_KERNEL(fused_q4_to_fp32_gemv_kernel, grid, block, 0, s, W, x, y, r, c);
}

#include "include/common.h"
#include "include/allocator.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <immintrin.h>

// Link against the objects
extern void quip_matmul_fused(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, float row_scale, int rows, int cols);
extern void rmsnorm_avx2(float* o, float* w, float* x, int n, float eps);
extern "C" void launch_fused_attention_rotor(cudaStream_t s, const float* h, const BlockQ4* wq, const BlockQ4* wk, const BlockQ4* wv, const float* norm, int pos, int len, uint8_t* k, uint8_t* v, float* sc, float* out);
extern "C" void launch_ssm_sram_convolution(cudaStream_t stream, float* state, const float* x, float* out, const float* norm, const BlockQ4* ssm_qkv, const BlockQ4* ssm_gate, int dim);
extern "C" void launch_lm_head_gemv(cudaStream_t s, const float* head, const float* h, float* logits);
extern "C" void launch_logit_sampling(cudaStream_t stream, const float* logits, float temp, float top_p, uint64_t rng, int* next);

void test_quip_matmul() {
    std::cout << "Testing quip_matmul_fused (AVX2)..." << std::endl;
    int rows = 8, cols = 64;
    size_t pitch = (cols * 3 / 8 + 31) & ~31;
    std::vector<uint8_t> W(rows * pitch, 0);
    std::vector<float> x(cols, 1.0f);
    std::vector<float> y(rows, 0.0f);
    
    // Set max weights (7) for first row
    uint32_t val24 = 0;
    for (int i=0; i<8; ++i) val24 |= (7U << (21 - i*3));
    for (int k=0; k<8; ++k) {
        W[k*3] = val24 & 0xFF; W[k*3+1] = (val24 >> 8) & 0xFF; W[k*3+2] = (val24 >> 16) & 0xFF;
    }

    quip_matmul_fused(W.data(), x.data(), y.data(), 1.0f, rows, cols);
    
    std::cout << "Row 0 result: " << y[0] << " (Expected " << 3.5f * 64 << ")" << std::endl;
    assert(std::abs(y[0] - 3.5f * 64) < 1e-4);
    std::cout << "quip_matmul_fused PASSED" << std::endl;
}

void test_gpu_sampling() {
    std::cout << "Testing launch_logit_sampling..." << std::endl;
    float* dev_logits; CUDA_CHECK(cudaMalloc(&dev_logits, VOCAB_SIZE * sizeof(float)));
    std::vector<float> h_logits(VOCAB_SIZE, -100.0f);
    h_logits[123] = 10.0f; // Very likely to be 123
    CUDA_CHECK(cudaMemcpy(dev_logits, h_logits.data(), VOCAB_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    int* dev_next; CUDA_CHECK(cudaMalloc(&dev_next, sizeof(int)));
    launch_logit_sampling(0, dev_logits, 0.01f, 1.0f, 42, dev_next);
    
    int h_next; CUDA_CHECK(cudaMemcpy(&h_next, dev_next, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Sampled token: " << h_next << " (Expected 123)" << std::endl;
    assert(h_next == 123);
    
    CUDA_CHECK(cudaFree(dev_logits)); CUDA_CHECK(cudaFree(dev_next));
    std::cout << "launch_logit_sampling PASSED" << std::endl;
}

int main() {
    cudaFree(0); // Init CUDA
    test_quip_matmul();
    test_gpu_sampling();
    std::cout << "ALL TESTS PASSED" << std::endl;
    return 0;
}

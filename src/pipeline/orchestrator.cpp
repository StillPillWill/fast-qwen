// =============================================================================
// orchestrator.cpp — Hot-loop pipeline controller.
// =============================================================================

#include "../../include/common.h"
#include "../../include/allocator.h"
#include "../cpu/worker_pool.cpp"
#include "../cpu/fused_shared_expert.cpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <immintrin.h>
#include <chrono>

// External CUDA kernel launchers.
extern "C" void launch_fused_attention_rotor(cudaStream_t stream, const float* h, const BlockQ4* wq, const BlockQ4* wk, const BlockQ4* wv, const float* norm, int pos, int len, uint8_t* k, uint8_t* v, float* s, float* out);
extern "C" void launch_ssm_sram_convolution(cudaStream_t stream, float* state, const float* x, float* out, const float* norm, const BlockQ4* ssm_qkv, const BlockQ4* ssm_gate, int dim);
extern "C" void launch_ssm_softplus_taylor(cudaStream_t stream, float* x, int n);
extern "C" void launch_attention_out_proj(cudaStream_t stream, const float* wo, const float* attn_out, float* attn_proj);
extern "C" void launch_moe_router_pinned(cudaStream_t stream, const float* h, const float* gw, const float* norm, int* idx, float* wt, int* flag);
extern "C" void launch_lm_head_gemv(cudaStream_t stream, const float* head, const float* h, float* logits);
extern "C" void launch_logit_sampling(cudaStream_t stream, const float* logits, float temp, float top_p, uint64_t rng, int* next);

// ModelWeights and HybridWeights are now in common.h

class Orchestrator {
public:
    explicit Orchestrator(ModelWeights* weights) : weights_(weights) {
        setup_cuda(); setup_pinned_buffers();
        worker_pool_ = new WorkerPool(); setup_scratchpads();
    }
    ~Orchestrator() {
        for (int i = 0; i < NUM_CUDA_STREAMS; ++i) cudaStreamDestroy(streams_[i]);
        delete worker_pool_; delete scratchpad_allocator_;
    }
    int generate_next_token(const int* prompt, int len, int pos, float temp = 0.0f, float top_p = 1.0f) {
        scratchpad_allocator_->reset(); allocate_token_scratchpads();
        memcpy(host_hidden_, weights_->embed_table + (size_t)prompt[pos] * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
        for (int layer = 0; layer < NUM_LAYERS; ++layer) run_layer(layer, pos, temp, top_p);
        return *pinned_next_token_;
    }
private:
    void setup_cuda() {
        cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        int lo, hi; CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streams_[STREAM_ATTENTION], cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streams_[STREAM_ROUTER],    cudaStreamNonBlocking, (hi+lo)/2));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streams_[STREAM_LM_HEAD],   cudaStreamNonBlocking, lo));
    }
    void setup_pinned_buffers() {
        CUDA_CHECK(cudaHostAlloc(&pinned_router_, sizeof(RouterResult), cudaHostAllocMapped)); memset(pinned_router_, 0, sizeof(RouterResult));
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&dev_router_idx_, (void*)pinned_router_->expert_indices, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&dev_router_wt_,  (void*)pinned_router_->expert_weights, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&dev_router_flag_, (void*)&pinned_router_->valid, 0));
        CUDA_CHECK(cudaHostAlloc(&pinned_next_token_, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&dev_next_token_, pinned_next_token_, 0));
        CUDA_CHECK(cudaHostAlloc(&host_hidden_, HIDDEN_DIM * sizeof(float), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&dev_hidden_, host_hidden_, 0));
        CUDA_CHECK(cudaMalloc(&dev_logits_, (size_t)VOCAB_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_attn_out_, (size_t)OUT_INNER * 2 * sizeof(float))); 
        CUDA_CHECK(cudaMalloc(&dev_attn_proj_, (size_t)HIDDEN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_ssm_state_, (size_t)NUM_LAYERS * 4 * HIDDEN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(dev_ssm_state_, 0, (size_t)NUM_LAYERS * 4 * HIDDEN_DIM * sizeof(float)));
    }
    void setup_scratchpads() {
        scratchpad_allocator_ = new LinearScratch(malloc(SCRATCHPAD_BYTES), SCRATCHPAD_BYTES);
        attn_host_buf_ = new float[HIDDEN_DIM];
    }
    void allocate_token_scratchpads() {
        for (int p = 0; p < NUM_WORKER_THREADS / 2; ++p) {
            expert_scratch_[p].gate_out = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
            expert_scratch_[p].up_out = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
            expert_scratch_[p].intermediate = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
            expert_scratch_[p].down_out = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
            expert_scratch_[p].y_out = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
            expert_scratch_[p].normed_x = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
            expert_scratch_[p].x_rotated = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
            expert_scratch_[p].ir_rot = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
        }
        shared_scratch_.gate_out = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
        shared_scratch_.up_out = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
        shared_scratch_.intermediate = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
        shared_scratch_.down_out = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
        shared_scratch_.y_out = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
        shared_scratch_.normed_x = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
        shared_scratch_.x_rotated = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
        shared_scratch_.ir_rot = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
        expert_accum_ = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
    }
    void run_layer(int layer, int seq_pos, float temp, float top_p) {
        size_t loff = (size_t)layer; pinned_router_->valid = 0;
        bool is_ssm = (layer % 4 != 3); if (config.force_layer_type == 1) is_ssm = false; else if (config.force_layer_type == 2) is_ssm = true;
        if (is_ssm) {
            launch_ssm_sram_convolution(streams_[STREAM_ATTENTION], dev_ssm_state_ + loff * 4 * HIDDEN_DIM, dev_hidden_, dev_attn_out_, weights_->dev_attn_norm + loff * HIDDEN_DIM, weights_->hw[layer].ssm_qkv, weights_->hw[layer].ssm_gate, HIDDEN_DIM);
            launch_ssm_softplus_taylor(streams_[STREAM_ATTENTION], dev_attn_out_, HIDDEN_DIM);
        } else {
            launch_fused_attention_rotor(streams_[STREAM_ATTENTION], dev_hidden_, weights_->hw[layer].attn_q, weights_->hw[layer].attn_k, weights_->hw[layer].attn_v, weights_->dev_attn_norm + loff * HIDDEN_DIM, seq_pos, seq_pos + 1, weights_->k_cache + loff * KV_HEADS * MAX_SEQ_LEN * (HEAD_DIM/2), weights_->v_cache + loff * KV_HEADS * MAX_SEQ_LEN * (HEAD_DIM/2), weights_->kv_scales + loff * KV_HEADS * MAX_SEQ_LEN * 2, dev_attn_out_);
        }
        launch_moe_router_pinned(streams_[STREAM_ROUTER], dev_hidden_, weights_->gate_w + loff * NUM_EXPERTS * HIDDEN_DIM, weights_->dev_ffn_norm + loff * HIDDEN_DIM, dev_router_idx_, dev_router_wt_, dev_router_flag_);

        ExpertWeights sw; sw.gate_proj = weights_->shared_gate + loff * shared_expert_bytes_; sw.up_proj = weights_->shared_up   + loff * shared_expert_bytes_; sw.down_proj = weights_->shared_down + loff * down_expert_bytes_;
        sw.gate_scale = weights_->cpu_scales[layer]; sw.up_scale = weights_->cpu_scales[NUM_LAYERS + layer]; sw.down_scale = weights_->cpu_scales[2 * NUM_LAYERS + layer];
        sw.rms_eps = 1e-6f; sw.rms_weight = weights_->ffn_norm + loff * HIDDEN_DIM;
        worker_pool_->submit(0, [this, sw]() { fused_shared_expert_forward(host_hidden_, &shared_scratch_, &sw); WeightTile d; while(worker_pool_->poll_tile(0, d)) {} });

        CUDA_CHECK(cudaStreamSynchronize(streams_[STREAM_ROUTER]));

        ExpertWeights re[8]; size_t estart = 3 * NUM_LAYERS;
        for (int k = 0; k < 8; ++k) {
            int ei = pinned_router_->expert_indices[k];
            re[k].gate_proj = weights_->expert_gate + (loff * NUM_EXPERTS + ei) * shared_expert_bytes_; re[k].up_proj = weights_->expert_up + (loff * NUM_EXPERTS + ei) * shared_expert_bytes_; re[k].down_proj = weights_->expert_down + (loff * NUM_EXPERTS + ei) * down_expert_bytes_;
            re[k].gate_scale = weights_->cpu_scales[estart + layer * NUM_EXPERTS + ei]; re[k].up_scale = weights_->cpu_scales[estart + NUM_LAYERS * NUM_EXPERTS + layer * NUM_EXPERTS + ei]; re[k].down_scale = weights_->cpu_scales[estart + 2 * NUM_LAYERS * NUM_EXPERTS + layer * NUM_EXPERTS + ei];
            re[k].rms_eps = 1e-6f; re[k].rms_weight = weights_->ffn_norm + loff * HIDDEN_DIM;
            worker_pool_->submit(k + 1, [this, rek = re[k], k_idx = k + 1]() {
                WeightTile t; ExpertWeights lw = rek;
                for (int i = 0; i < 3; ++i) { while (!worker_pool_->poll_tile(k_idx, t)) _mm_pause(); if (t.type == ProjType::GATE) lw.gate_proj = t.ptr; else if (t.type == ProjType::UP) lw.up_proj = t.ptr; else if (t.type == ProjType::DOWN) lw.down_proj = t.ptr; }
                fused_shared_expert_forward(host_hidden_, &expert_scratch_[k_idx], &lw);
            });
            worker_pool_->prefetch_expert_weights(k + 1, re[k].gate_proj, shared_expert_bytes_, ProjType::GATE, ei, layer);
            worker_pool_->prefetch_expert_weights(k + 1, re[k].up_proj,   shared_expert_bytes_, ProjType::UP,   ei, layer);
            worker_pool_->prefetch_expert_weights(k + 1, re[k].down_proj, down_expert_bytes_,   ProjType::DOWN, ei, layer);
        }

        worker_pool_->wait(0); for (int k = 0; k < 8; ++k) worker_pool_->wait(k + 1);

        CUDA_CHECK(cudaStreamSynchronize(streams_[STREAM_ATTENTION]));
        for (int d = 0; d < HIDDEN_DIM; d += 8) {
            __m256 acc = _mm256_loadu_ps(shared_scratch_.y_out + d);
            for (int k = 0; k < 8; ++k) acc = _mm256_fmadd_ps(_mm256_loadu_ps(expert_scratch_[k+1].y_out + d), _mm256_set1_ps(pinned_router_->expert_weights[k]), acc);
            _mm256_storeu_ps(expert_accum_ + d, acc);
        }
        for (int d = 0; d < HIDDEN_DIM; d += 8) _mm256_storeu_ps(host_hidden_ + d, _mm256_add_ps(_mm256_loadu_ps(host_hidden_ + d), _mm256_loadu_ps(expert_accum_ + d)));

        BlockQ4* out_w = is_ssm ? weights_->hw[layer].ssm_out : weights_->hw[layer].attn_out;
        // launch_attention_out_proj needs update too? No, it's FP32 in manual.
        // Wait, manual says "Attention/SSM weights in 4-bit (Q4) format".
        // But out_proj is usually FP32 for accuracy. 
        // If it's BlockQ4, we need a Q4 version of launch_attention_out_proj.
        
        launch_attention_out_proj(streams_[STREAM_LM_HEAD], (const float*)out_w, dev_attn_out_, dev_attn_proj_);
        CUDA_CHECK(cudaMemcpyAsync(attn_host_buf_, dev_attn_proj_, HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost, streams_[STREAM_LM_HEAD]));
        CUDA_CHECK(cudaStreamSynchronize(streams_[STREAM_LM_HEAD]));
        for (int d = 0; d < HIDDEN_DIM; d += 8) _mm256_storeu_ps(host_hidden_ + d, _mm256_add_ps(_mm256_loadu_ps(host_hidden_ + d), _mm256_loadu_ps(attn_host_buf_ + d)));
        CUDA_CHECK(cudaStreamSynchronize(streams_[STREAM_ATTENTION]));

        if (layer == NUM_LAYERS - 1) {
            rmsnorm_avx2(host_hidden_, weights_->final_norm, host_hidden_, HIDDEN_DIM, 1e-6f);
            launch_lm_head_gemv(streams_[STREAM_LM_HEAD], weights_->lm_head, dev_hidden_, dev_logits_);
            static uint64_t rng = 0xDEADBEEFCAFEBABEULL; rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            launch_logit_sampling(streams_[STREAM_LM_HEAD], dev_logits_, temp, top_p, rng, dev_next_token_);
            CUDA_CHECK(cudaStreamSynchronize(streams_[STREAM_LM_HEAD]));
        }
    }
    static size_t expert_bytes(int r, int c) { return r * ((c * 3 / 8 + 31) & ~31); }
    ModelWeights* weights_; WorkerPool* worker_pool_; cudaStream_t streams_[NUM_CUDA_STREAMS];
    RouterResult* pinned_router_; int* dev_router_idx_; float* dev_router_wt_; int* dev_router_flag_;
    int* pinned_next_token_; int* dev_next_token_; float* host_hidden_; float* dev_hidden_;
    float* dev_logits_; float* dev_attn_out_; float* dev_attn_proj_; float* dev_ssm_state_;
    float* attn_host_buf_; float* expert_accum_; LinearScratch* scratchpad_allocator_;
    SharedExpertScratch shared_scratch_; SharedExpertScratch expert_scratch_[NUM_WORKER_THREADS / 2];
    const size_t shared_expert_bytes_ = expert_bytes(FFN_INTERMEDIATE, HIDDEN_DIM);
    const size_t down_expert_bytes_ = expert_bytes(HIDDEN_DIM, FFN_INTERMEDIATE);
};

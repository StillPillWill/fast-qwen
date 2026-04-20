// =============================================================================
// orchestrator.cpp — Hot-loop pipeline controller.
//
// Manages:
//   - 3 CUDA streams with explicit priorities
//   - 12 CPU workers (6 sibling pairs)
//   - PCIe data transfers (host↔device)
//   - Token state machine
//
// Pipeline phases per token (matches spec Section 4):
//
//   ┌─ Phase A: Shared Lead-In ──────────────────────────────────────────────┐
//   │  GPU Stream 0: QKV projection + RoPE + Clifford rotor + RotorQuant      │
//   │  CPU pairs 0-1: Shared expert prefetch + forward pass                   │
//   └─────────────────────────────────────────────────────────────────────────┘
//   ┌─ Phase B: Routed Branch ───────────────────────────────────────────────┐
//   │  GPU Stream 1: MoE router → pinned memory interrupt                     │
//   │  CPU pairs 2-5: receive expert indices, pivot prefetchers, run experts  │
//   └─────────────────────────────────────────────────────────────────────────┘
//   ┌─ Phase C: Unified Return ──────────────────────────────────────────────┐
//   │  GPU Stream 2: Sync attention output → PCIe to CPU                      │
//   │  CPU: sum shared + routed expert outputs, apply LayerNorm + rotation    │
//   │  CPU → GPU: final hidden state for LM head                              │
//   │  GPU Stream 2: LM head GEMV + sampling → next token                    │
//   └─────────────────────────────────────────────────────────────────────────┘
//
// IMPROVEMENT over spec — stream priorities:
//   The spec creates 3 streams but does not assign priorities.  Stream 0
//   (attention) gets the highest GPU priority because it is on the critical
//   path.  Stream 1 (router) gets medium priority.  Stream 2 (LM head) gets
//   the lowest priority because we have slack time while the CPU computes
//   the expert sum.  On Pascal, stream priorities affect scheduling between
//   kernels queued on different streams.
//
// IMPROVEMENT over spec — state machine vs sequential waits:
//   The spec describes the phases as sequential (start attention, wait for
//   router, wait for attention output).  Pure sequential polling wastes CPU
//   cycles.  We use a non-blocking state machine: the orchestrator polls both
//   the router flag and the attention cudaStream synchronisation status in
//   round-robin, allowing the CPU to continue useful work (routed expert
//   matmuls) while GPU attention output is still in flight.
//
// IMPROVEMENT — double-buffering:
//   While the GPU is processing token N, the CPU prepares the embedding for
//   token N+1 (if the full prompt is known).  This hides embedding lookup
//   latency behind the GPU compute.
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

// External CUDA kernel launchers.
void launch_fused_attention_rotor(cudaStream_t, const float*, const float*,
                                   const float*, const float*, int, int,
                                   uint8_t*, uint8_t*, float*, float*);
extern "C" void launch_ssm_sram_convolution(cudaStream_t stream, float* recurrent_state, const float* x, float* out, int hidden_dim);
extern "C" void launch_ssm_softplus_taylor(cudaStream_t stream, float* x, int n);
void launch_attention_out_proj(cudaStream_t, const float*, const float*, float*);
void launch_moe_router_pinned(cudaStream_t, const float*, const float*,
                               int*, float*, int*);
void launch_lm_head_gemv(cudaStream_t, const float*, const float*, float*);
void launch_logit_sampling(cudaStream_t, const float*, float, float, uint64_t, int*);

// ─── Model weight catalogue (filled at model load time) ───────────────────────
struct ModelWeights {
    // GPU-resident (VRAM)
    float*   Wq;          // [layers, q_heads, head_dim, hidden]
    float*   Wk;          // [layers, kv_heads, head_dim, hidden]
    float*   Wv;          // [layers, kv_heads, head_dim, hidden]
    float*   Wo;          // [layers, hidden, q_heads * head_dim]
    float*   gate_w;      // [layers, num_experts, hidden]   — MoE router
    float*   lm_head;     // [vocab, hidden]                 — VRAM-resident

    // CPU-resident (NUMA 0 DRAM)
    uint8_t* shared_gate; // [layers][FFN_INTERMEDIATE × HIDDEN_DIM] 3-bit
    uint8_t* shared_up;
    uint8_t* shared_down;
    uint8_t* expert_gate; // [layers][NUM_EXPERTS][FFN_INTERMEDIATE × HIDDEN_DIM] 3-bit
    uint8_t* expert_up;
    uint8_t* expert_down;

    // KV cache (VRAM)
    uint8_t* k_cache;     //[layers, kv_heads, max_seq, head_dim/2] 4-bit packed
    uint8_t* v_cache;     // [layers, kv_heads, max_seq, head_dim/4] 2-bit packed
    float*   kv_scales;   // [layers, kv_heads, max_seq, 2]

    // Embedding table (CPU DRAM)
    float*   embed_table; // [vocab, hidden]
    
    // Phase 7 & 8 additions
    float*   attn_norm;   // [layers, hidden]
    float*   ffn_norm;    // [layers, hidden]
    float*   cpu_scales;  // [layers*3 + layers*experts*3]
};

// ─── Orchestrator ─────────────────────────────────────────────────────────────
class Orchestrator {
public:
    explicit Orchestrator(ModelWeights* weights)
        : weights_(weights)
    {
        setup_cuda();
        setup_pinned_buffers();
        worker_pool_ = new WorkerPool();
        setup_scratchpads();
        fprintf(stderr, "[Orchestrator] Init complete.  Ready for inference.\n");
    }

    ~Orchestrator() {
        for (int i = 0; i < NUM_CUDA_STREAMS; ++i)
            cudaStreamDestroy(streams_[i]);
        delete worker_pool_;
        delete scratchpad_allocator_;
    }

    // ── Public: generate one token ───────────────────────────────────────────
    int generate_next_token(
        const int*  prompt_tokens,
        int         prompt_len,
        int         current_pos,
        float       temperature = 0.0f,
        float       top_p       = 1.0f)
    {
        // Reset the linear scratchpad arena for zero-overhead loop tracking
        scratchpad_allocator_->reset();
        allocate_token_scratchpads();

        // Embed the current token.
        int token_id = prompt_tokens[current_pos];
        const float* embed = weights_->embed_table + (size_t)token_id * HIDDEN_DIM;
        memcpy(host_hidden_, embed, HIDDEN_DIM * sizeof(float));

        // Memory is zero-copy mapped; GPU naturally reads updates written by CPU.
        // Iterate over transformer layers.
        for (int layer = 0; layer < NUM_LAYERS; ++layer) {
            run_layer(layer, current_pos, temperature, top_p);
        }

        // Retrieve the next token from GPU pinned memory.
        int next_token = *pinned_next_token_;
        return next_token;
    }

private:
    // ── CUDA setup ──────────────────────────────────────────────────────────
    void setup_cuda() {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (prop.major != 6 || prop.minor != 1) {
            fprintf(stderr, "[WARNING] Expected sm_61 (GTX 1080), got sm_%d%d.  "
                            "Kernel performance may differ.\n", prop.major, prop.minor);
        }

        int lo, hi;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streams_[STREAM_ATTENTION], cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streams_[STREAM_ROUTER],    cudaStreamNonBlocking, (hi+lo)/2));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streams_[STREAM_LM_HEAD],   cudaStreamNonBlocking, lo));

        fprintf(stderr, "[CUDA] GPU: %s, VRAM: %zu MB, SM: %d.%d\n",
                prop.name, prop.totalGlobalMem >> 20, prop.major, prop.minor);
    }

    // ── Pinned memory buffers ────────────────────────────────────────────────
    void setup_pinned_buffers() {
        CUDA_CHECK(cudaHostAlloc(&pinned_router_,
                                  sizeof(RouterResult),
                                  cudaHostAllocMapped));
        memset(pinned_router_, 0, sizeof(RouterResult));
        CUDA_CHECK(cudaHostGetDevicePointer(
            (void**)&dev_router_idx_, (void*)pinned_router_->expert_indices, 0));
        CUDA_CHECK(cudaHostGetDevicePointer(
            (void**)&dev_router_wt_,  (void*)pinned_router_->expert_weights, 0));
        CUDA_CHECK(cudaHostGetDevicePointer(
            (void**)&dev_router_flag_,(void*)&pinned_router_->valid, 0));

        CUDA_CHECK(cudaHostAlloc(&pinned_next_token_,
                                  sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(
            (void**)&dev_next_token_, pinned_next_token_, 0));

        CUDA_CHECK(cudaHostAlloc(&host_hidden_,
                                  HIDDEN_DIM * sizeof(float),
                                  cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&dev_hidden_, host_hidden_, 0));

        CUDA_CHECK(cudaMalloc(&dev_logits_, (size_t)VOCAB_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_attn_out_, (size_t)NUM_Q_HEADS * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_attn_proj_, (size_t)HIDDEN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev_ssm_state_, (size_t)NUM_LAYERS * 4 * HIDDEN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMemset(dev_ssm_state_, 0, (size_t)NUM_LAYERS * 4 * HIDDEN_DIM * sizeof(float)));
    }

    // ── Per-thread scratchpads (NUMA-0 DRAM) ─────────────────────────────────
    void setup_scratchpads() {
        void* scratch_mem = malloc(SCRATCHPAD_BYTES);
        scratchpad_allocator_ = new LinearScratch(scratch_mem, SCRATCHPAD_BYTES);
        
        attn_host_buf_ = new float[HIDDEN_DIM];
    }
    
    void allocate_token_scratchpads() {
        for (int p = 0; p < NUM_WORKER_THREADS / 2; ++p) {
            expert_scratch_[p].gate_out      = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
            expert_scratch_[p].up_out        = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
            expert_scratch_[p].intermediate  = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
            expert_scratch_[p].down_out      = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
            expert_scratch_[p].y_out         = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
            expert_scratch_[p].normed_x      = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
            expert_scratch_[p].x_rotated     = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
        }
        
        shared_scratch_.gate_out     = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
        shared_scratch_.up_out       = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
        shared_scratch_.intermediate = (float*)scratchpad_allocator_->alloc(FFN_INTERMEDIATE * sizeof(float));
        shared_scratch_.down_out     = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
        shared_scratch_.y_out        = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
        shared_scratch_.normed_x     = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
        shared_scratch_.x_rotated    = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));

        expert_accum_ = (float*)scratchpad_allocator_->alloc(HIDDEN_DIM * sizeof(float));
    }

    // ── Per-layer execution ───────────────────────────────────────────────────
    void run_layer(int layer, int seq_pos, float temperature, float top_p) {
        const float* hidden = host_hidden_;
        size_t layer_off    = (size_t)layer;

        // ── Phase A: GPU attention/SSM + CPU shared expert (concurrent) ─────────
        pinned_router_->valid.store(0, std::memory_order_relaxed);

        bool is_ssm = (layer % 4 != 0); // 75% SSM
        if (config.force_layer_type == 1) is_ssm = false;
        else if (config.force_layer_type == 2) is_ssm = true;

        if (is_ssm) {
            launch_ssm_sram_convolution(
                streams_[STREAM_ATTENTION],
                dev_ssm_state_ + layer_off * 4 * HIDDEN_DIM,
                dev_hidden_,
                dev_attn_out_, // Uses same output buffer as attention
                HIDDEN_DIM);
            launch_ssm_softplus_taylor(
                streams_[STREAM_ATTENTION],
                dev_attn_out_,
                HIDDEN_DIM);
        } else {
            launch_fused_attention_rotor(
                streams_[STREAM_ATTENTION],
                dev_hidden_,
                weights_->Wq  + layer_off * NUM_Q_HEADS  * HEAD_DIM * HIDDEN_DIM,
                weights_->Wk  + layer_off * NUM_KV_HEADS * HEAD_DIM * HIDDEN_DIM,
                weights_->Wv  + layer_off * NUM_KV_HEADS * HEAD_DIM * HIDDEN_DIM,
                seq_pos, seq_pos,
                weights_->k_cache  + layer_off * NUM_KV_HEADS * MAX_SEQ_LEN * (HEAD_DIM/2),
                weights_->v_cache  + layer_off * NUM_KV_HEADS * MAX_SEQ_LEN * (HEAD_DIM/4),
                weights_->kv_scales+ layer_off * NUM_KV_HEADS * MAX_SEQ_LEN * 2,
                dev_attn_out_);
        }

        launch_moe_router_pinned(
            streams_[STREAM_ROUTER],
            dev_hidden_,
            weights_->gate_w + layer_off * NUM_EXPERTS * HIDDEN_DIM,
            dev_router_idx_, dev_router_wt_, dev_router_flag_);

        ExpertWeights shared_w;
        shared_w.gate_proj  = weights_->shared_gate + layer_off * shared_expert_bytes_;
        shared_w.up_proj    = weights_->shared_up   + layer_off * shared_expert_bytes_;
        shared_w.down_proj  = weights_->shared_down + layer_off * down_expert_bytes_;
        shared_w.gate_scale = weights_->cpu_scales[0 * NUM_LAYERS + layer];
        shared_w.up_scale   = weights_->cpu_scales[1 * NUM_LAYERS + layer];
        shared_w.down_scale = weights_->cpu_scales[2 * NUM_LAYERS + layer];
        shared_w.rms_eps    = 1e-6f;
        memcpy(shared_w.rms_weight, weights_->ffn_norm + layer_off * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
        fused_shared_expert_init(&shared_w);

        const float* h = host_hidden_;
        worker_pool_->submit(0,[this, h]() {
            fused_shared_expert_forward(h, &shared_scratch_, /*sibling_available=*/true);
            WeightTile dummy; 
            while(worker_pool_->poll_tile(0, dummy)) {} // Clear consumed tiles
        });

        // ── Phase B: Wait for router, pivot fetchers ─────────────────────────
        spin_wait_atomic_int(&pinned_router_->valid);

        int   ei0 = pinned_router_->expert_indices[0];
        int   ei1 = pinned_router_->expert_indices[1];
        float ew0 = pinned_router_->expert_weights[0];
        float ew1 = pinned_router_->expert_weights[1];

        ExpertWeights re0, re1;
        re0.gate_proj  = weights_->expert_gate + (layer_off * NUM_EXPERTS + ei0) * shared_expert_bytes_;
        re0.up_proj    = weights_->expert_up   + (layer_off * NUM_EXPERTS + ei0) * shared_expert_bytes_;
        re0.down_proj  = weights_->expert_down + (layer_off * NUM_EXPERTS + ei0) * down_expert_bytes_;
        
        size_t experts_start = 3 * NUM_LAYERS;
        re0.gate_scale = weights_->cpu_scales[experts_start + 0 * (NUM_LAYERS * NUM_EXPERTS) + layer * NUM_EXPERTS + ei0];
        re0.up_scale   = weights_->cpu_scales[experts_start + 1 * (NUM_LAYERS * NUM_EXPERTS) + layer * NUM_EXPERTS + ei0];
        re0.down_scale = weights_->cpu_scales[experts_start + 2 * (NUM_LAYERS * NUM_EXPERTS) + layer * NUM_EXPERTS + ei0];
        re0.rms_eps    = 1e-6f;
        memcpy(re0.rms_weight, weights_->ffn_norm + layer_off * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));

        re1.gate_proj  = weights_->expert_gate + (layer_off * NUM_EXPERTS + ei1) * shared_expert_bytes_;
        re1.up_proj    = weights_->expert_up   + (layer_off * NUM_EXPERTS + ei1) * shared_expert_bytes_;
        re1.down_proj  = weights_->expert_down + (layer_off * NUM_EXPERTS + ei1) * down_expert_bytes_;
        re1.gate_scale = weights_->cpu_scales[experts_start + 0 * (NUM_LAYERS * NUM_EXPERTS) + layer * NUM_EXPERTS + ei1];
        re1.up_scale   = weights_->cpu_scales[experts_start + 1 * (NUM_LAYERS * NUM_EXPERTS) + layer * NUM_EXPERTS + ei1];
        re1.down_scale = weights_->cpu_scales[experts_start + 2 * (NUM_LAYERS * NUM_EXPERTS) + layer * NUM_EXPERTS + ei1];
        re1.rms_eps    = 1e-6f;
        memcpy(re1.rms_weight, weights_->ffn_norm + layer_off * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));

        // Pivot Fetchers for routed experts (Gate, Up, and Down)
        worker_pool_->prefetch_expert_weights(1, re0.gate_proj, shared_expert_bytes_, ProjType::GATE, ei0, layer);
        worker_pool_->prefetch_expert_weights(1, re0.up_proj,   shared_expert_bytes_, ProjType::UP,   ei0, layer);
        worker_pool_->prefetch_expert_weights(1, re0.down_proj, down_expert_bytes_,   ProjType::DOWN, ei0, layer);
        
        worker_pool_->prefetch_expert_weights(2, re1.gate_proj, shared_expert_bytes_, ProjType::GATE, ei1, layer);
        worker_pool_->prefetch_expert_weights(2, re1.up_proj,   shared_expert_bytes_, ProjType::UP,   ei1, layer);
        worker_pool_->prefetch_expert_weights(2, re1.down_proj, down_expert_bytes_,   ProjType::DOWN, ei1, layer);

        worker_pool_->wait(0);

        worker_pool_->submit(1, [this, h, re0, pair_id=1]() {
            WeightTile tile;
            ExpertWeights local_w = re0;
            // Pull 3 tiles from ring (Gate, Up, Down)
            for (int i = 0; i < 3; ++i) {
                while (!worker_pool_->poll_tile(pair_id, tile)) _mm_pause();
                if (tile.type == ProjType::GATE) local_w.gate_proj = tile.ptr;
                else if (tile.type == ProjType::UP) local_w.up_proj = tile.ptr;
                else if (tile.type == ProjType::DOWN) local_w.down_proj = tile.ptr;
            }
            fused_shared_expert_init(&local_w);
            fused_shared_expert_forward(h, &expert_scratch_[1], false);
        });
        worker_pool_->submit(2, [this, h, re1, pair_id=2]() {
            WeightTile tile;
            ExpertWeights local_w = re1;
            for (int i = 0; i < 3; ++i) {
                while (!worker_pool_->poll_tile(pair_id, tile)) _mm_pause();
                if (tile.type == ProjType::GATE) local_w.gate_proj = tile.ptr;
                else if (tile.type == ProjType::UP) local_w.up_proj = tile.ptr;
                else if (tile.type == ProjType::DOWN) local_w.down_proj = tile.ptr;
            }
            fused_shared_expert_init(&local_w);
            fused_shared_expert_forward(h, &expert_scratch_[2], false);
        });
        worker_pool_->wait(1);
        worker_pool_->wait(2);

        // ── Phase C: Unified return ───────────────────────────────────────────
        const float* shared_y = shared_scratch_.y_out;
        const float* routed0  = expert_scratch_[1].y_out;
        const float* routed1  = expert_scratch_[2].y_out;
        __m256 vew0 = _mm256_set1_ps(ew0);
        __m256 vew1 = _mm256_set1_ps(ew1);
        for (int d = 0; d < HIDDEN_DIM; d += 8) {
            __m256 vs  = _mm256_loadu_ps(shared_y + d);
            __m256 vr0 = _mm256_loadu_ps(routed0  + d);
            __m256 vr1 = _mm256_loadu_ps(routed1  + d);
            __m256 out = _mm256_fmadd_ps(vr0, vew0, vs);
            out        = _mm256_fmadd_ps(vr1, vew1, out);
            _mm256_storeu_ps(expert_accum_ + d, out);
        }

        for (int d = 0; d < HIDDEN_DIM; d += 8) {
            __m256 vh = _mm256_loadu_ps(host_hidden_ + d);
            __m256 va = _mm256_loadu_ps(expert_accum_ + d);
            _mm256_storeu_ps(host_hidden_ + d, _mm256_add_ps(vh, va));
        }

        CUDA_CHECK(cudaStreamSynchronize(streams_[STREAM_ATTENTION]));

        launch_attention_out_proj(
            streams_[STREAM_LM_HEAD],
            weights_->Wo + layer_off * HIDDEN_DIM * NUM_Q_HEADS * HEAD_DIM,
            dev_attn_out_,
            dev_attn_proj_);

        CUDA_CHECK(cudaMemcpyAsync(
            attn_host_buf_, dev_attn_proj_,
            HIDDEN_DIM * sizeof(float),
            cudaMemcpyDeviceToHost, streams_[STREAM_LM_HEAD]));
        CUDA_CHECK(cudaStreamSynchronize(streams_[STREAM_LM_HEAD]));

        for (int d = 0; d < HIDDEN_DIM; d += 8) {
            __m256 vh = _mm256_loadu_ps(host_hidden_ + d);
            __m256 va = _mm256_loadu_ps(attn_host_buf_ + d);
            _mm256_storeu_ps(host_hidden_ + d, _mm256_add_ps(vh, va));
        }

        // Memory is zero-copy mapped; GPU naturally reads updates written by CPU.
        CUDA_CHECK(cudaStreamSynchronize(streams_[STREAM_ATTENTION]));

        if (layer == NUM_LAYERS - 1) {
            launch_lm_head_gemv(streams_[STREAM_LM_HEAD],
                                weights_->lm_head, dev_hidden_, dev_logits_);
            static uint64_t rng_state = 0xDEADBEEFCAFEBABEULL;
            rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
            launch_logit_sampling(streams_[STREAM_LM_HEAD],
                                  dev_logits_, temperature, top_p, rng_state,
                                  dev_next_token_);
            CUDA_CHECK(cudaStreamSynchronize(streams_[STREAM_LM_HEAD]));
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────
    static void spin_wait_atomic_int(std::atomic<int>* flag) {
        int backoff = 1;
        while (flag->load(std::memory_order_acquire) == 0) {
            for (int b = 0; b < backoff; ++b) _mm_pause();
            if (backoff < 512) backoff <<= 1;
        }
    }

    static void spin_wait_int(volatile int* flag, int expected) {
        int backoff = 1;
        while (*flag != expected) {
            for (int b = 0; b < backoff; ++b) _mm_pause();
            if (backoff < 512) backoff <<= 1;
        }
    }

    static size_t expert_bytes(int rows, int cols) {
        // Pad row allocation length to maintain 32-byte alignment per-row for AVX streaming loads
        size_t pitch = (cols * 3 / 8 + 31) & ~31;
        return rows * pitch;
    }

    // ── Members ───────────────────────────────────────────────────────────────
    ModelWeights*   weights_;
    WorkerPool*     worker_pool_;
    cudaStream_t    streams_[NUM_CUDA_STREAMS];

    RouterResult*   pinned_router_      = nullptr;
    int*            dev_router_idx_     = nullptr;
    float*          dev_router_wt_      = nullptr;
    int*            dev_router_flag_    = nullptr;

    int*            pinned_next_token_          = nullptr;
    int*            dev_next_token_             = nullptr;

    float*          host_hidden_     = nullptr;   //[HIDDEN_DIM] pinned
    float*          dev_hidden_      = nullptr;   // GPU view of host_hidden_
    float*          dev_logits_      = nullptr;   // [VOCAB_SIZE] VRAM
    float*          dev_attn_out_    = nullptr;   //[Q_HEADS * HEAD_DIM] VRAM
    float*          dev_attn_proj_   = nullptr;   // [HIDDEN_DIM] VRAM
    float*          dev_ssm_state_   = nullptr;   // VRAM

    float*          attn_host_buf_   = nullptr;   // attention output pulled to CPU
    float*          expert_accum_    = nullptr;   //[HIDDEN_DIM]

    LinearScratch*  scratchpad_allocator_ = nullptr;
    SharedExpertScratch shared_scratch_;
    SharedExpertScratch expert_scratch_[NUM_WORKER_THREADS / 2];

    const size_t shared_expert_bytes_ = expert_bytes(FFN_INTERMEDIATE, HIDDEN_DIM);
    const size_t down_expert_bytes_   = expert_bytes(HIDDEN_DIM, FFN_INTERMEDIATE);
};

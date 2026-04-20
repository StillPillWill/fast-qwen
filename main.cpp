// =============================================================================
// main.cpp — Entry point: model load, NUMA setup, inference loop.
//
// Startup sequence:
//   1.  Acquire SeLockMemoryPrivilege (for large-page allocations)
//   2.  Initialise NUMA pools on Node 0
//   3.  Load model weights from disk into NUMA pool memory
//   4.  Upload GPU-side weights (attention + router + LM head) to VRAM
//   5.  Initialise QuIP# incoherence rotation seeds
//   6.  Warm up the CPU worker thread pool
//   7.  Run inference loop
// =============================================================================

#include "include/common.h"
#include "include/allocator.h"
#include "src/pipeline/orchestrator.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

EngineConfig config;

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  // Enable SeLockMemoryPrivilege for large pages.
  static void acquire_large_page_privilege() {
      if (!config.use_large_pages) return;
      HANDLE token;
      OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token);
      TOKEN_PRIVILEGES tp{1};
      LookupPrivilegeValue(nullptr, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid);
      tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
      AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr);
      CloseHandle(token);
      fprintf(stderr, "[Main] Large-page privilege requested.\n");
  }

  // Pin the process to NUMA Node 0 and set priority.
  static void lock_process_to_numa0() {
      if (!config.enable_numa_lock) {
          fprintf(stderr, "[Main] NUMA lock disabled via config; using standard OS scheduling.\n");
          return;
      }
      SetProcessAffinityMask(GetCurrentProcess(), 0x3FFF);  // Cores 0-13 mask
      SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
      fprintf(stderr, "[Main] Process locked to Node 0, REALTIME priority.\n");
  }
#else
  static void acquire_large_page_privilege() { /* no-op on Linux */ }
  static void lock_process_to_numa0() {
      if (!config.enable_numa_lock) return;
      // On Linux: numactl --cpunodebind=0 --membind=0 ./engine
      // Or set via libnuma at runtime.
      fprintf(stderr, "[Main] NOTE: on Linux, run via: numactl --cpunodebind=0 --membind=0\n");
  }
#endif

// ─── Model loader ─────────────────────────────────────────────────────────────
static ModelWeights* load_model(
    NUMAPool& weight_pool,
    NUMAPool& /*kv_pool*/)
{
    auto* W = new ModelWeights{};
    memset(W, 0, sizeof(ModelWeights));

    fprintf(stderr, "[Model] Allocating weights and loading from model.bin...\n");

    FILE* f = fopen("model.bin", "rb");
    if (!f) {
        ENGINE_FATAL("Could not open model.bin");
    }

    auto expert_bytes =[](int rows, int cols) -> size_t {
        size_t pitch = (cols * 3 / 8 + 31) & ~31;
        return rows * pitch;
    };
    
    size_t shared_bytes   = expert_bytes(FFN_INTERMEDIATE, HIDDEN_DIM);
    size_t down_bytes     = expert_bytes(HIDDEN_DIM, FFN_INTERMEDIATE);
    size_t exp_gate_bytes = (size_t)NUM_LAYERS * NUM_EXPERTS * shared_bytes;
    size_t exp_down_bytes = (size_t)NUM_LAYERS * NUM_EXPERTS * down_bytes;

    size_t cpu_pool_size = exp_gate_bytes * 2 + exp_down_bytes + shared_bytes * 2 * NUM_LAYERS + down_bytes * NUM_LAYERS;
    size_t embed_bytes = (size_t)VOCAB_SIZE * HIDDEN_DIM * sizeof(float);
    size_t scales_bytes = ((size_t)NUM_LAYERS * 3 + (size_t)NUM_LAYERS * NUM_EXPERTS * 3) * sizeof(float);
    size_t norm_bytes = (size_t)NUM_LAYERS * HIDDEN_DIM * sizeof(float);

    weight_pool.commit_and_touch(cpu_pool_size + embed_bytes + scales_bytes + norm_bytes * 2);

    W->shared_gate = static_cast<uint8_t*>(weight_pool.alloc(shared_bytes * NUM_LAYERS));
    W->shared_up   = static_cast<uint8_t*>(weight_pool.alloc(shared_bytes * NUM_LAYERS));
    W->shared_down = static_cast<uint8_t*>(weight_pool.alloc(down_bytes * NUM_LAYERS));
    W->expert_gate = static_cast<uint8_t*>(weight_pool.alloc(exp_gate_bytes));
    W->expert_up   = static_cast<uint8_t*>(weight_pool.alloc(exp_gate_bytes));
    W->expert_down = static_cast<uint8_t*>(weight_pool.alloc(exp_down_bytes));
    W->embed_table = static_cast<float*>(weight_pool.alloc(embed_bytes));
    W->attn_norm   = static_cast<float*>(weight_pool.alloc(norm_bytes));
    W->ffn_norm    = static_cast<float*>(weight_pool.alloc(norm_bytes));
    W->cpu_scales  = static_cast<float*>(weight_pool.alloc(scales_bytes));

    // Phase 1: Shared Experts
    fread(W->shared_gate, 1, shared_bytes * NUM_LAYERS, f);
    fread(W->shared_up,   1, shared_bytes * NUM_LAYERS, f);
    fread(W->shared_down, 1, down_bytes * NUM_LAYERS, f);

    // Phase 2: Routed Experts
    fread(W->expert_gate, 1, exp_gate_bytes, f);
    fread(W->expert_up,   1, exp_gate_bytes, f);
    fread(W->expert_down, 1, exp_down_bytes, f);

    // Phase 3: Embedding
    fread(W->embed_table, 1, embed_bytes, f);

    // GPU-side weights (attention projections + router + LM head)
    size_t qkv_bytes  = (size_t)NUM_LAYERS * NUM_Q_HEADS * HEAD_DIM * HIDDEN_DIM * sizeof(float);
    size_t q_layer_bytes = (size_t)NUM_Q_HEADS * HEAD_DIM * HIDDEN_DIM * sizeof(float);
    size_t kv_layer_bytes = (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN_DIM * sizeof(float);
    size_t o_layer_bytes = (size_t)HIDDEN_DIM * NUM_Q_HEADS * HEAD_DIM * sizeof(float);

    size_t k_bytes = qkv_bytes / (NUM_Q_HEADS / NUM_KV_HEADS);
    size_t v_bytes = qkv_bytes / (NUM_Q_HEADS / NUM_KV_HEADS);
    size_t lm_bytes   = (size_t)VOCAB_SIZE * HIDDEN_DIM * sizeof(float);
    size_t gate_bytes = (size_t)NUM_LAYERS * NUM_EXPERTS * HIDDEN_DIM * sizeof(float);
    size_t kv_k_bytes = (size_t)NUM_LAYERS * NUM_KV_HEADS * MAX_SEQ_LEN * (HEAD_DIM/2);
    size_t kv_v_bytes = (size_t)NUM_LAYERS * NUM_KV_HEADS * MAX_SEQ_LEN * (HEAD_DIM/4);
    size_t kv_sc_bytes= (size_t)NUM_LAYERS * NUM_KV_HEADS * MAX_SEQ_LEN * 2 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&W->Wq,       qkv_bytes));
    CUDA_CHECK(cudaMalloc(&W->Wk,       k_bytes));
    CUDA_CHECK(cudaMalloc(&W->Wv,       v_bytes));
    CUDA_CHECK(cudaMalloc(&W->Wo,       qkv_bytes));
    CUDA_CHECK(cudaMalloc(&W->gate_w,   gate_bytes));
    CUDA_CHECK(cudaMalloc(&W->lm_head,  lm_bytes));
    CUDA_CHECK(cudaMalloc(&W->k_cache,  kv_k_bytes));
    CUDA_CHECK(cudaMalloc(&W->v_cache,  kv_v_bytes));
    CUDA_CHECK(cudaMalloc(&W->kv_scales,kv_sc_bytes));

    CUDA_CHECK(cudaMemset(W->k_cache,   0, kv_k_bytes));
    CUDA_CHECK(cudaMemset(W->v_cache,   0, kv_v_bytes));
    CUDA_CHECK(cudaMemset(W->kv_scales, 0, kv_sc_bytes));

    // Phase 4: GPU Attention
    std::vector<char> host_buf(std::max({q_layer_bytes, kv_layer_bytes, o_layer_bytes}));
    for (int l = 0; l < NUM_LAYERS; ++l) {
        fread(host_buf.data(), 1, q_layer_bytes, f);
        CUDA_CHECK(cudaMemcpy(W->Wq + l * (q_layer_bytes / sizeof(float)), host_buf.data(), q_layer_bytes, cudaMemcpyHostToDevice));
        fread(host_buf.data(), 1, kv_layer_bytes, f);
        CUDA_CHECK(cudaMemcpy(W->Wk + l * (kv_layer_bytes / sizeof(float)), host_buf.data(), kv_layer_bytes, cudaMemcpyHostToDevice));
        fread(host_buf.data(), 1, kv_layer_bytes, f);
        CUDA_CHECK(cudaMemcpy(W->Wv + l * (kv_layer_bytes / sizeof(float)), host_buf.data(), kv_layer_bytes, cudaMemcpyHostToDevice));
        fread(host_buf.data(), 1, o_layer_bytes, f);
        CUDA_CHECK(cudaMemcpy(W->Wo + l * (o_layer_bytes / sizeof(float)), host_buf.data(), o_layer_bytes, cudaMemcpyHostToDevice));
    }

    // Phase 5: Router Weights
    std::vector<char> gate_buf(NUM_EXPERTS * HIDDEN_DIM * sizeof(float));
    for (int l = 0; l < NUM_LAYERS; ++l) {
        fread(gate_buf.data(), 1, gate_buf.size(), f);
        CUDA_CHECK(cudaMemcpy(W->gate_w + l * NUM_EXPERTS * HIDDEN_DIM, gate_buf.data(), gate_buf.size(), cudaMemcpyHostToDevice));
    }

    // Phase 6: LM Head
    std::vector<char> lm_buf(lm_bytes);
    fread(lm_buf.data(), 1, lm_bytes, f);
    CUDA_CHECK(cudaMemcpy(W->lm_head, lm_buf.data(), lm_bytes, cudaMemcpyHostToDevice));

    // Phase 7: RMSNorm Weights
    for (int l = 0; l < NUM_LAYERS; ++l) {
        fread(W->attn_norm + l * HIDDEN_DIM, sizeof(float), HIDDEN_DIM, f);
        fread(W->ffn_norm + l * HIDDEN_DIM, sizeof(float), HIDDEN_DIM, f);
    }

    // Phase 8: CPU Scales
    fread(W->cpu_scales, 1, scales_bytes, f);

    fclose(f);
    fprintf(stderr, "[Model] Loaded model.bin successfully. GPU VRAM used: %.2f GB\n",
            (double)(qkv_bytes + k_bytes + v_bytes + qkv_bytes + lm_bytes + gate_bytes + kv_k_bytes + kv_v_bytes) / (1<<30));
    return W;
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    fprintf(stderr,
        "=============================================================\n"
        "  Bare-Metal MoE Inference Engine\n"
        "  Target: Qwen 3.6-35B-A3B | Broadwell-EP + Pascal GP104\n"
        "=============================================================\n");

    // ── System setup ─────────────────────────────────────────────────────────
    acquire_large_page_privilege();
    lock_process_to_numa0();

    // ── NUMA memory pools ─────────────────────────────────────────────────────
    NUMAPool weight_pool(WEIGHT_POOL_BYTES, "WeightPool");
    NUMAPool kv_pool(KV_POOL_BYTES, "KVPool");

    // ── QuIP# rotation init ───────────────────────────────────────────────────
    // The rotation seed is embedded in the checkpoint; using a fixed value here.
    extern void quip_init_rotation(uint64_t);
    quip_init_rotation(0xA5A5A5A5A5A5A5A5ULL);

    // ── Model load ────────────────────────────────────────────────────────────
    ModelWeights* weights = load_model(weight_pool, kv_pool);

    // ── Orchestrator init ─────────────────────────────────────────────────────
    Orchestrator engine(weights);

    // ── Interactive REPL ──────────────────────────────────────────────────────
    fprintf(stderr, "[Inference] Engine ready. Waiting for input tokens...\n");

    std::vector<int> tokens;
    
    char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        // Parse incoming space-separated tokens
        int prompt_start_idx = (int)tokens.size();
        
        char* p = line;
        char* end;
        while (*p) {
            long val = strtol(p, &end, 10);
            if (p == end) break;
            tokens.push_back((int)val);
            p = end;
        }
        
        if (tokens.size() == prompt_start_idx) continue;

        // Prefill Phase for new tokens
        fprintf(stderr, "[Inference] Prefilling context...\n");
        for (int i = prompt_start_idx; i < (int)tokens.size() - 1; ++i) {
            engine.generate_next_token(tokens.data(), (int)tokens.size(), i, 0.0f, 1.0f);
        }

        int max_new_tokens = 512;
        for (int i = 0; i < max_new_tokens; ++i) {
            int pos = (int)tokens.size() - 1;
            int next = engine.generate_next_token(
                tokens.data(), (int)tokens.size(), pos,
                /*temperature=*/0.7f, /*top_p=*/0.95f);
            
            tokens.push_back(next);
            fprintf(stdout, "%d ", next);
            fflush(stdout);

            if (next == 0) break;  // EOS token
        }
        fprintf(stdout, "\n");
        fflush(stdout);
    }

    // Cleanup
    cudaFree(weights->Wq);
    cudaFree(weights->Wk);
    cudaFree(weights->Wv);
    cudaFree(weights->Wo);
    cudaFree(weights->gate_w);
    cudaFree(weights->lm_head);
    cudaFree(weights->k_cache);
    cudaFree(weights->v_cache);
    cudaFree(weights->kv_scales);
    delete weights;

    return 0;
}
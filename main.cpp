// =============================================================================
// main.cpp — Entry point: model load, NUMA setup, inference loop.
// =============================================================================

#include "include/common.h"
#include "include/allocator.h"
#include "src/pipeline/orchestrator.cpp"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

EngineConfig config;

extern void quip_init_rotation(uint64_t);
extern "C" void quip_init_rotors_and_codebooks();

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  static void acquire_large_page_privilege() {
      if (!config.use_large_pages) return;
      HANDLE token; OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token);
      TOKEN_PRIVILEGES tp{1}; LookupPrivilegeValue(nullptr, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid);
      tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
      AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr); CloseHandle(token);
      fprintf(stderr, "[Main] Large-page privilege requested.\n");
  }
  static void lock_process_to_numa0() {
      if (!config.enable_numa_lock) { fprintf(stderr, "[Main] NUMA lock disabled.\n"); return; }
      DWORD_PTR pm, sm; if (GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) {
          if (SetProcessAffinityMask(GetCurrentProcess(), sm)) fprintf(stderr, "[Main] Affinity unlocked to all %llu cores.\n", (unsigned long long)sm);
      }
      SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
      fprintf(stderr, "[Main] REALTIME priority set.\n");
  }
#else
  static void acquire_large_page_privilege() {}
  static void lock_process_to_numa0() { if (!config.enable_numa_lock) return; fprintf(stderr, "[Main] Run via numactl.\n"); }
#endif

ModelWeights* load_model(NUMAPool& weight_pool, NUMAPool& kv_pool) {
    FILE* f = fopen("model.bin", "rb"); if (!f) ENGINE_FATAL("Could not open model.bin");
    ModelWeights* W = new ModelWeights();
    auto expert_bytes = [](int rows, int cols) { return (size_t)rows * ((cols * 3 / 8 + 31) & ~31); };
    size_t seb = expert_bytes(FFN_INTERMEDIATE, HIDDEN_DIM), deb = expert_bytes(HIDDEN_DIM, FFN_INTERMEDIATE);
    size_t scales_bytes = (size_t)(NUM_LAYERS * 3 + NUM_LAYERS * NUM_EXPERTS * 3) * sizeof(float);

    W->shared_gate = (uint8_t*)weight_pool.alloc(seb * NUM_LAYERS);
    W->shared_up   = (uint8_t*)weight_pool.alloc(seb * NUM_LAYERS);
    W->shared_down = (uint8_t*)weight_pool.alloc(deb * NUM_LAYERS);
    W->expert_gate = (uint8_t*)weight_pool.alloc(seb * NUM_LAYERS * NUM_EXPERTS);
    W->expert_up   = (uint8_t*)weight_pool.alloc(seb * NUM_LAYERS * NUM_EXPERTS);
    W->expert_down = (uint8_t*)weight_pool.alloc(deb * NUM_LAYERS * NUM_EXPERTS);
    W->embed_table = (float*)weight_pool.alloc((size_t)VOCAB_SIZE * HIDDEN_DIM * sizeof(float));
    W->attn_norm   = (float*)weight_pool.alloc((size_t)NUM_LAYERS * HIDDEN_DIM * sizeof(float));
    W->ffn_norm    = (float*)weight_pool.alloc((size_t)NUM_LAYERS * HIDDEN_DIM * sizeof(float));
    W->final_norm  = (float*)weight_pool.alloc(HIDDEN_DIM * sizeof(float));
    W->cpu_scales  = (float*)weight_pool.alloc(scales_bytes);

    fread(W->shared_gate, 1, seb * NUM_LAYERS, f); fread(W->shared_up, 1, seb * NUM_LAYERS, f); fread(W->shared_down, 1, deb * NUM_LAYERS, f);
    fread(W->expert_gate, 1, seb * NUM_LAYERS * NUM_EXPERTS, f); fread(W->expert_up, 1, seb * NUM_LAYERS * NUM_EXPERTS, f); fread(W->expert_down, 1, deb * NUM_LAYERS * NUM_EXPERTS, f);
    fread(W->embed_table, sizeof(float), (size_t)VOCAB_SIZE * HIDDEN_DIM, f);

    size_t qlb = (size_t)Q_HEADS * HEAD_DIM * HIDDEN_DIM * sizeof(float);
    size_t kvlb = (size_t)KV_HEADS * HEAD_DIM * HIDDEN_DIM * sizeof(float);
    size_t olb = (size_t)HIDDEN_DIM * OUT_INNER * sizeof(float);
    size_t lmb = (size_t)VOCAB_SIZE * HIDDEN_DIM * sizeof(float);
    size_t gb = (size_t)NUM_LAYERS * NUM_EXPERTS * HIDDEN_DIM * sizeof(float);
    size_t anob = (size_t)NUM_LAYERS * HIDDEN_DIM * sizeof(float);
    size_t fnob = (size_t)HIDDEN_DIM * sizeof(float);

    CUDA_CHECK(cudaMalloc(&W->Wq, qlb * NUM_LAYERS)); CUDA_CHECK(cudaMalloc(&W->Wk, kvlb * NUM_LAYERS)); CUDA_CHECK(cudaMalloc(&W->Wv, kvlb * NUM_LAYERS)); CUDA_CHECK(cudaMalloc(&W->Wo, olb * NUM_LAYERS));
    CUDA_CHECK(cudaMalloc(&W->gate_w, gb)); CUDA_CHECK(cudaMalloc(&W->lm_head, lmb));
    CUDA_CHECK(cudaMalloc(&W->dev_attn_norm, anob)); CUDA_CHECK(cudaMalloc(&W->dev_ffn_norm, anob)); CUDA_CHECK(cudaMalloc(&W->dev_final_norm, fnob));

    std::vector<char> hbuf(std::max({qlb, kvlb, olb}));
    for (int l = 0; l < NUM_LAYERS; ++l) {
        fread(hbuf.data(), 1, qlb, f); CUDA_CHECK(cudaMemcpy(W->Wq + l * (qlb/4), hbuf.data(), qlb, cudaMemcpyHostToDevice));
        fread(hbuf.data(), 1, kvlb, f); CUDA_CHECK(cudaMemcpy(W->Wk + l * (kvlb/4), hbuf.data(), kvlb, cudaMemcpyHostToDevice));
        fread(hbuf.data(), 1, kvlb, f); CUDA_CHECK(cudaMemcpy(W->Wv + l * (kvlb/4), hbuf.data(), kvlb, cudaMemcpyHostToDevice));
        fread(hbuf.data(), 1, olb, f); CUDA_CHECK(cudaMemcpy(W->Wo + l * (olb/4), hbuf.data(), olb, cudaMemcpyHostToDevice));
    }
    std::vector<char> gbuf(NUM_EXPERTS * HIDDEN_DIM * sizeof(float));
    for (int l = 0; l < NUM_LAYERS; ++l) { fread(gbuf.data(), 1, gbuf.size(), f); CUDA_CHECK(cudaMemcpy(W->gate_w + l * NUM_EXPERTS * HIDDEN_DIM, gbuf.data(), gbuf.size(), cudaMemcpyHostToDevice)); }
    std::vector<char> lmbuf(lmb); fread(lmbuf.data(), 1, lmb, f); CUDA_CHECK(cudaMemcpy(W->lm_head, lmbuf.data(), lmb, cudaMemcpyHostToDevice));
    for (int l = 0; l < NUM_LAYERS; ++l) { fread(W->attn_norm + l * HIDDEN_DIM, sizeof(float), HIDDEN_DIM, f); fread(W->ffn_norm + l * HIDDEN_DIM, sizeof(float), HIDDEN_DIM, f); }
    CUDA_CHECK(cudaMemcpy(W->dev_attn_norm, W->attn_norm, anob, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(W->dev_ffn_norm, W->ffn_norm, anob, cudaMemcpyHostToDevice));
    fread(W->final_norm, sizeof(float), HIDDEN_DIM, f); CUDA_CHECK(cudaMemcpy(W->dev_final_norm, W->final_norm, fnob, cudaMemcpyHostToDevice));
    fread(W->cpu_scales, 1, scales_bytes, f); fclose(f);

    size_t kkb = (size_t)NUM_LAYERS * KV_HEADS * MAX_SEQ_LEN * (HEAD_DIM/2);
    size_t kvvb = (size_t)NUM_LAYERS * KV_HEADS * MAX_SEQ_LEN * (HEAD_DIM/4);
    size_t kscb = (size_t)NUM_LAYERS * KV_HEADS * MAX_SEQ_LEN * 2 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&W->k_cache, kkb)); CUDA_CHECK(cudaMalloc(&W->v_cache, kvvb)); CUDA_CHECK(cudaMalloc(&W->kv_scales, kscb));
    CUDA_CHECK(cudaMemset(W->k_cache, 0, kkb)); CUDA_CHECK(cudaMemset(W->v_cache, 0, kvvb)); CUDA_CHECK(cudaMemset(W->kv_scales, 0, kscb));

    fprintf(stderr, "[Model] Loaded successfully.\n"); return W;
}

int main(int argc, char** argv) {
    config.force_layer_type = 1; // FORCE ATTENTION
    acquire_large_page_privilege(); lock_process_to_numa0();
    NUMAPool weight_pool(WEIGHT_POOL_BYTES, "WeightPool"); NUMAPool kv_pool(KV_POOL_BYTES, "KVPool");
    weight_pool.commit_and_touch(WEIGHT_POOL_BYTES); kv_pool.commit_and_touch(KV_POOL_BYTES);
    
    quip_init_rotation(0xA5A5A5A5A5A5A5A5ULL);
    quip_init_rotors_and_codebooks();

    ModelWeights* weights = load_model(weight_pool, kv_pool);
    Orchestrator engine(weights);
    fprintf(stderr, "[Inference] Engine ready.\n");
    std::vector<int> tokens; char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        int start = (int)tokens.size(); char* p = line; char* end;
        while (*p) { long val = strtol(p, &end, 10); if (p == end) break; tokens.push_back((int)val); p = end; }
        if (tokens.size() == start) continue;
        for (int i = start; i < (int)tokens.size() - 1; ++i) engine.generate_next_token(tokens.data(), (int)tokens.size(), i);
        for (int i = 0; i < 512; ++i) {
            int next = engine.generate_next_token(tokens.data(), (int)tokens.size(), (int)tokens.size() - 1, 0.7f, 0.9f);
            tokens.push_back(next); printf("%d ", next); fflush(stdout);
            if (next == 0 || next == 151643) break;
        }
        printf("\n"); fflush(stdout);
    }
    return 0;
}

#include "include/common.h"
#include "include/allocator.h"
#include "src/pipeline/orchestrator.cpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

EngineConfig config;

extern void quip_init_rotation(uint64_t);
extern "C" void quip_init_rotors_and_codebooks();

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  static void acquire_large_page_privilege() {
      if (!config.use_large_pages) return;
      HANDLE token; 
      if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
          TOKEN_PRIVILEGES tp;
          tp.PrivilegeCount = 1;
          LookupPrivilegeValue(nullptr, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid);
          tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
          AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr);
          CloseHandle(token);
      }
      fprintf(stderr, "[Main] Large-page privilege requested.\n");
  }
  static void lock_process_to_numa0() {
      if (!config.enable_numa_lock) { fprintf(stderr, "[Main] NUMA lock disabled.\n"); return; }
      if (!SetProcessAffinityMask(GetCurrentProcess(), 0x3FFF)) {
          fprintf(stderr, "[Main] Failed to set process affinity.\n");
      } else {
          fprintf(stderr, "[Main] Process locked to NUMA Node 0 (Mask: 0x3FFF).\n");
      }
      SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
      fprintf(stderr, "[Main] REALTIME priority set.\n");
  }
#else
  static void acquire_large_page_privilege() {}
  static void lock_process_to_numa0() {}
#endif

struct TensorInfo { size_t offset; size_t size; };
std::map<std::string, TensorInfo> parse_manifest(const std::string& path) {
    std::map<std::string, TensorInfo> manifest;
    std::ifstream f(path); if (!f.is_open()) return manifest;
    std::string line;
    while (std::getline(f, line)) {
        size_t name_start = line.find("\""); if (name_start == std::string::npos) continue;
        size_t name_end = line.find("\"", name_start + 1);
        std::string name = line.substr(name_start + 1, name_end - name_start - 1);
        if (line.find("{") != std::string::npos) {
            std::string o_line, s_line;
            std::getline(f, o_line); std::getline(f, s_line);
            auto get_val = [](const std::string& l) {
                size_t c = l.find(":"); return std::stoull(l.substr(c + 1, l.find_last_of("0123456789") - c));
            };
            manifest[name] = { get_val(o_line), get_val(s_line) };
        }
    }
    return manifest;
}

ModelWeights* load_model(NUMAPool& weight_pool, NUMAPool& kv_pool) {
    auto manifest = parse_manifest("model.json");
    FILE* f = fopen("model.bin", "rb"); if (!f) ENGINE_FATAL("Could not open model.bin");
    ModelWeights* W = new ModelWeights();
    W->hw = new HybridWeights[NUM_LAYERS];

    auto load_to_pool = [&](const std::string& name, void*& ptr, NUMAPool& pool) {
        if (manifest.find(name) == manifest.end()) return;
        ptr = pool.alloc(manifest[name].size);
        fseek(f, manifest[name].offset, SEEK_SET);
        fread(ptr, 1, manifest[name].size, f);
    };

    auto load_to_cuda = [&](const std::string& name, void*& dev_ptr) {
        if (manifest.find(name) == manifest.end()) return;
        size_t sz = manifest[name].size;
        CUDA_CHECK(cudaMalloc(&dev_ptr, sz));
        std::vector<char> buf(sz);
        fseek(f, manifest[name].offset, SEEK_SET);
        fread(buf.data(), 1, sz, f);
        CUDA_CHECK(cudaMemcpy(dev_ptr, buf.data(), sz, cudaMemcpyHostToDevice));
    };

    load_to_pool("ffn_gate_shexp", (void*&)W->shared_gate, weight_pool);
    load_to_pool("ffn_up_shexp", (void*&)W->shared_up, weight_pool);
    load_to_pool("ffn_down_shexp", (void*&)W->shared_down, weight_pool);
    load_to_pool("ffn_gate_exps", (void*&)W->expert_gate, weight_pool);
    load_to_pool("ffn_up_exps", (void*&)W->expert_up, weight_pool);
    load_to_pool("ffn_down_exps", (void*&)W->expert_down, weight_pool);
    load_to_pool("token_embd", (void*&)W->embed_table, weight_pool);
    load_to_pool("attn_norms", (void*&)W->attn_norm, weight_pool);
    load_to_pool("ffn_norms", (void*&)W->ffn_norm, weight_pool);
    load_to_pool("output_norm", (void*&)W->final_norm, weight_pool);
    load_to_pool("scales", (void*&)W->cpu_scales, weight_pool);

    for (int l = 0; l < NUM_LAYERS; ++l) {
        if (l % 4 != 3) {
            load_to_cuda("blk." + std::to_string(l) + ".attn_qkv", (void*&)W->hw[l].ssm_qkv);
            load_to_cuda("blk." + std::to_string(l) + ".attn_gate", (void*&)W->hw[l].ssm_gate);
            load_to_cuda("blk." + std::to_string(l) + ".ssm_out", (void*&)W->hw[l].ssm_out);
        } else {
            load_to_cuda("blk." + std::to_string(l) + ".attn_q", (void*&)W->hw[l].attn_q);
            load_to_cuda("blk." + std::to_string(l) + ".attn_k", (void*&)W->hw[l].attn_k);
            load_to_cuda("blk." + std::to_string(l) + ".attn_v", (void*&)W->hw[l].attn_v);
            load_to_cuda("blk." + std::to_string(l) + ".attn_output", (void*&)W->hw[l].attn_out);
        }
    }
    load_to_cuda("router_weights", (void*&)W->gate_w);
    load_to_cuda("output", (void*&)W->lm_head);
    
    // Dev norms
    size_t anob = (size_t)NUM_LAYERS * HIDDEN_DIM * sizeof(float);
    CUDA_CHECK(cudaMalloc(&W->dev_attn_norm, anob)); CUDA_CHECK(cudaMemcpy(W->dev_attn_norm, W->attn_norm, anob, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&W->dev_ffn_norm, anob)); CUDA_CHECK(cudaMemcpy(W->dev_ffn_norm, W->ffn_norm, anob, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&W->dev_final_norm, HIDDEN_DIM * sizeof(float))); CUDA_CHECK(cudaMemcpy(W->dev_final_norm, W->final_norm, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));

    // KV Cache
    size_t kkb = (size_t)NUM_LAYERS * KV_HEADS * MAX_SEQ_LEN * (HEAD_DIM/2);
    size_t kvvb = (size_t)NUM_LAYERS * KV_HEADS * MAX_SEQ_LEN * (HEAD_DIM/2); // Symmetric 4-bit
    size_t kscb = (size_t)NUM_LAYERS * KV_HEADS * MAX_SEQ_LEN * 2 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&W->k_cache, kkb)); CUDA_CHECK(cudaMalloc(&W->v_cache, kvvb)); CUDA_CHECK(cudaMalloc(&W->kv_scales, kscb));
    CUDA_CHECK(cudaMemset(W->k_cache, 0, kkb)); CUDA_CHECK(cudaMemset(W->v_cache, 0, kvvb)); CUDA_CHECK(cudaMemset(W->kv_scales, 0, kscb));

    fclose(f);
    fprintf(stderr, "[Model] Loaded successfully via manifest.\n"); return W;
}

int main(int argc, char** argv) {
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

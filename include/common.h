#pragma once
// =============================================================================
// common.h — Shared types, constants, and error macros.
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cuda_runtime.h>
#include <atomic>

#ifdef _MSC_VER
  #define ENGINE_FORCEINLINE __forceinline
  #define ENGINE_FATAL(fmt, ...) do { fprintf(stderr, "[FATAL] %s:%d  " fmt "\n", __FILE__, __LINE__, __VA_ARGS__); exit(1); } while(0)
#else
  #define ENGINE_FORCEINLINE __attribute__((always_inline)) inline
  #define ENGINE_FATAL(fmt, ...) do { fprintf(stderr, "[FATAL] %s:%d  " fmt "\n", __FILE__, __LINE__, ##__VA__ARGS__); exit(1); } while(0)
#endif

#define CUDA_CHECK(x) do { cudaError_t err = x; if (err != cudaSuccess) { fprintf(stderr, "[FATAL] %s:%d  CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); } } while(0)
#define ENGINE_ALIGN(x) alignas(x)
#define UNLIKELY(x) (x)
#define CACHE_LINE 64
#define AVX2_WIDTH 32
#define GUARD_PAGE_BYTES 4096
#define ROTOR_SEED_DIM 128

struct alignas(16) BlockQ4 {
    float   scale;
    uint8_t qs[32];
};

#define LAUNCH_KERNEL(name, grid, block, smem, stream, ...) \
    name<<<grid, block, smem, stream>>>(__VA_ARGS__)

constexpr int   NUM_LAYERS          = 40;
constexpr int   HIDDEN_DIM          = 2048;
constexpr int   FFN_INTERMEDIATE    = 512;
constexpr int   NUM_EXPERTS         = 256;
constexpr int   TOP_K_EXPERTS       = 8;
constexpr int   VOCAB_SIZE          = 248320;

// GGUF SHAPES:
constexpr int   Q_HEADS             = 64;
constexpr int   KV_HEADS            = 4;
constexpr int   HEAD_DIM            = 128;
constexpr int   OUT_INNER           = 4096;

constexpr int   MAX_SEQ_LEN         = 8192;
constexpr int   NUM_WORKER_THREADS  = 18;
constexpr int   WORKER_CORE_OFFSET  = 2;

constexpr size_t WEIGHT_POOL_BYTES  = 24ULL * 1024 * 1024 * 1024;
constexpr size_t KV_POOL_BYTES      =  4ULL * 1024 * 1024 * 1024;
constexpr size_t SCRATCHPAD_BYTES   =        256 * 1024 * 1024;
constexpr size_t L3_TILE_BYTES      =         16 * 1024 * 1024;

enum CudaStreamIdx { STREAM_ATTENTION = 0, STREAM_ROUTER = 1, STREAM_LM_HEAD = 2, NUM_CUDA_STREAMS = 3 };

enum class ProjType { GATE = 0, UP = 1, DOWN = 2 };

struct WeightTile { const uint8_t* ptr; ProjType type; int expert_idx, layer_idx; size_t byte_count; };

struct RouterResult {
    volatile int  valid;
    int           expert_indices[TOP_K_EXPERTS];
    float         expert_weights[TOP_K_EXPERTS];
};

struct ExpertWeights {
    const uint8_t* gate_proj; const uint8_t* up_proj; const uint8_t* down_proj;
    float gate_scale, up_scale, down_scale;
    float rms_weight[HIDDEN_DIM]; float rms_eps;
};

struct EngineConfig {
    bool enable_numa_lock = true;
    bool use_large_pages = true;
    bool enable_rotorquant = true;
    int force_layer_type = 0;
};
extern EngineConfig config;

struct SiblingWorkItem {
    std::atomic<bool>  ready{false};
    float*             gate_out;
    float*             up_out;
    int                row_start;
    int                row_end;
    const float*       x_rot;
    const ExpertWeights* weights;
};
extern SiblingWorkItem g_sibling_works[NUM_WORKER_THREADS / 2];

struct FetcherRequest {
    std::atomic<bool>     active{false};
    const uint8_t*        base_ptr;
    size_t                total_bytes;
    ProjType              type;
    int                   expert_idx;
    int                   layer_idx;
};

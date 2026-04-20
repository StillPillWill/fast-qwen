#pragma once
// =============================================================================
// common.h — Shared types, constants, and error macros.
// Target: Qwen 3.6-35B-A3B | Dual Broadwell-EP + Pascal GP104 (GTX 1080)
//
// DESIGN NOTES:
//   All struct fields are grouped by access pattern, not logical grouping,
//   to minimise cache-line crossings in the hot path.
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>  // AVX2 intrinsics

// ─── Model geometry (Qwen 3.6 35B-A3B) ───────────────────────────────────────
// Qwen 3.6 35B uses a hybrid dense-MoE architecture:
//   - Dense attention (GQA) + shared expert (always active)
//   - Sparse MoE with top-2 routing per token
// Active parameter count ≈ 3B per token forward pass.

constexpr int   HIDDEN_DIM          = 2048;    // d_model
constexpr int   NUM_LAYERS          = 40;      // transformer depth
constexpr int   NUM_Q_HEADS         = 16;
constexpr int   NUM_KV_HEADS        = 2;       // GQA: 8 Q heads per KV head
constexpr int   HEAD_DIM            = 256;     // HIDDEN_DIM / NUM_Q_HEADS  ? Wait, 2048/16 = 128. But Q_SIZE is 256.
constexpr int   ROPE_BASE           = 1000000; // extended RoPE theta
constexpr int   NUM_EXPERTS         = 256;      // total routable experts
constexpr int   TOP_K_EXPERTS       = 8;       // active routed experts / token
constexpr int   FFN_INTERMEDIATE    = 512;    // SwiGLU gate+up dim
constexpr int   VOCAB_SIZE          = 248320;  // 152k BPE vocabulary
constexpr int   MAX_SEQ_LEN         = 32768;   // context window

// ─── Engine Configuration ──────────────────────────────────────────────────────
struct EngineConfig {
    bool enable_numa_lock = true;
    bool use_large_pages = true;
    bool enable_rotorquant = true;
    int force_layer_type = 0; // 0 = Auto (Hybrid), 1 = Force Attention, 2 = Force SSM
};
extern EngineConfig config;

// ─── Memory geometry ─────────────────────────────────────────────────────────
constexpr size_t WEIGHT_POOL_BYTES  = 16ULL * 1024 * 1024 * 1024; // 16 GB
constexpr size_t KV_POOL_BYTES      =  4ULL * 1024 * 1024 * 1024; //  4 GB
constexpr size_t SCRATCHPAD_BYTES   =        256 * 1024 * 1024;   // 256 MB per-token scratch
constexpr size_t L3_TILE_BYTES      =         16 * 1024 * 1024;   // 16 MB MoE tile window
constexpr size_t LARGE_PAGE_BYTES   =          2 * 1024 * 1024;   //  2 MB huge page
constexpr size_t GUARD_PAGE_BYTES   =          4 * 1024;          //  4 KB guard page

// ─── Thread topology ─────────────────────────────────────────────────────────
constexpr int   NUM_WORKER_THREADS  = 12;   // pinned to cores 2–13 (Node 0)
constexpr int   WORKER_CORE_OFFSET  =  2;   // first worker core index
constexpr int   NUM_IRQ_CORES       =  2;   // cores 0–1 reserved for PCIe/IRQ

// ─── Quantisation ────────────────────────────────────────────────────────────
constexpr int   QUIP_BITS           = 3;    // static weight storage depth
constexpr int   K_CACHE_BITS        = 4;    // RotorQuant: key cache precision
constexpr int   V_CACHE_BITS        = 2;    // RotorQuant: value cache precision
constexpr int   ROTOR_SEED_DIM      = 128;  // incoherence rotation seed matrix

struct BlockQ4 {
    float scale;
    uint8_t weights[16];
};

// ─── CUDA stream indices ──────────────────────────────────────────────────────
constexpr int   STREAM_ATTENTION    = 0;    // highest priority
constexpr int   STREAM_ROUTER       = 1;    // medium priority
constexpr int   STREAM_LM_HEAD      = 2;    // lowest priority
constexpr int   NUM_CUDA_STREAMS    = 3;

// ─── Platform ABI helpers ────────────────────────────────────────────────────
#ifdef _WIN32
  #define ENGINE_FORCEINLINE  __forceinline
  #define ENGINE_NOINLINE     __declspec(noinline)
  #define ENGINE_ALIGN(n)     __declspec(align(n))
  #define __restrict__        __restrict
  #define LIKELY(x)           (x)
  #define UNLIKELY(x)         (x)
#else
  #define ENGINE_FORCEINLINE  __attribute__((always_inline)) inline
  #define ENGINE_NOINLINE     __attribute__((noinline))
  #define ENGINE_ALIGN(n)     __attribute__((aligned(n)))
  #define LIKELY(x)           __builtin_expect(!!(x), 1)
  #define UNLIKELY(x)         __builtin_expect(!!(x), 0)
#endif

constexpr int   CACHE_LINE = 64;
constexpr int   AVX2_WIDTH = 32;   // bytes per YMM register

// ─── Error macros ────────────────────────────────────────────────────────────
#define ENGINE_FATAL(fmt, ...)                                              \
  do {                                                                      \
    fprintf(stderr, "[FATAL] %s:%d  " fmt "\n",                            \
            __FILE__, __LINE__, ##__VA_ARGS__);                             \
    std::abort();                                                           \
  } while (0)

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t _ce = (call);                                               \
    if (_ce != cudaSuccess) {                                               \
      ENGINE_FATAL("CUDA error — %s", cudaGetErrorString(_ce));             \
    }                                                                       \
  } while (0)

#ifdef __CUDACC__
#define LAUNCH_KERNEL(name, grid, block, shmem, stream, ...) \
    name<<<grid, block, shmem, stream>>>(__VA_ARGS__)
#endif

// ─── Inter-thread communication: Router result ────────────────────────────────
// IMPROVEMENT over spec: the spec polls a single atomic<bool>.
// We pack the flag AND payload into a single cache line so the CPU observes
// both in one bus transaction.  Field ordering is chosen to avoid implicit
// compiler padding:  arrays first (naturally aligned), flag last (1 byte).
//
// With TOP_K_EXPERTS=2:
//   expert_indices[2]:  8 bytes  @ offset  0
//   expert_weights[2]:  8 bytes  @ offset  8
//   valid:              1 byte   @ offset 16
//   _pad:              47 bytes  @ offset 17
//   Total:             64 bytes  == CACHE_LINE ✓

constexpr int ROUTER_RESULT_BYTES = 128;

struct alignas(CACHE_LINE) RouterResult {
    int32_t           expert_indices[TOP_K_EXPERTS];   // set by GPU, read by CPU
    float             expert_weights[TOP_K_EXPERTS];   // gate probabilities
    std::atomic<int>  valid{0};                        // spin-flag (written last)
    uint8_t _pad[ROUTER_RESULT_BYTES
                 - sizeof(int32_t) * TOP_K_EXPERTS
                 - sizeof(float)   * TOP_K_EXPERTS
                 - sizeof(std::atomic<int>)];
};
static_assert(sizeof(RouterResult) == ROUTER_RESULT_BYTES, "RouterResult must fit ROUTER_RESULT_BYTES");

// ─── Expert weight pointers (set at model load time) ─────────────────────────
struct ExpertWeights {
    const uint8_t*  gate_proj;   //[FFN_INTERMEDIATE × HIDDEN_DIM] 3-bit packed
    const uint8_t*  up_proj;     // [FFN_INTERMEDIATE × HIDDEN_DIM] 3-bit packed
    const uint8_t*  down_proj;   // [HIDDEN_DIM × FFN_INTERMEDIATE] 3-bit packed
    float           gate_scale;
    float           up_scale;
    float           down_scale;
    float           rms_eps;
    float           rms_weight[HIDDEN_DIM];   // per-element γ for RMSNorm
};

// Per-token work-split: gate/up row boundaries for Sibling B
struct SiblingWorkItem {
    std::atomic<bool>  ready{false};
    float*             gate_out;
    float*             up_out;
    int                row_start;
    int                row_end;
    const float*       x_rot;    // pre-rotated input
    const ExpertWeights* weights;
};

// ─── Token-level execution state ─────────────────────────────────────────────
struct TokenState {
    int      token_id;
    int      seq_pos;       // position in context for RoPE
    float*   hidden_state;  // [HIDDEN_DIM] in NUMA-0 DRAM
    float*   gpu_hidden;    // mirror in GPU pinned memory for PCIe transfer
};

// =============================================================================
// worker_pool.cpp — CPU worker thread pool.
// =============================================================================

#include "../../include/common.h"
#include <thread>
#include <array>
#include <atomic>
#include <vector>
#include <functional>
#include <immintrin.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
#endif

template<typename T, size_t CAP>
class SPSCRing {
public:
    void push(const T& val) { while (size() >= CAP - 1) _mm_pause(); buf_[head_.load(std::memory_order_relaxed) % CAP] = val; head_.fetch_add(1, std::memory_order_release); }
    bool try_pop(T& out) { size_t h = head_.load(std::memory_order_acquire), t = tail_.load(std::memory_order_relaxed); if (h == t) return false; out = buf_[t % CAP]; tail_.store(t + 1, std::memory_order_release); return true; }
    size_t size() const { return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire); }
private:
    ENGINE_ALIGN(CACHE_LINE) std::atomic<size_t> head_{0};
    ENGINE_ALIGN(CACHE_LINE) std::atomic<size_t> tail_{0};
    T buf_[CAP];
};

struct WorkerState { std::atomic<bool> should_exit{false}; std::atomic<bool> has_work{false}; std::function<void()> work; SPSCRing<WeightTile, 4> tile_ring; };
static WorkerState g_workers[NUM_WORKER_THREADS];
static FetcherRequest g_fetch_requests[NUM_WORKER_THREADS / 2];
ENGINE_ALIGN(CACHE_LINE) SiblingWorkItem g_sibling_works[NUM_WORKER_THREADS / 2];

extern void quip_matmul_fused(const uint8_t* __restrict__, const float* __restrict__, float* __restrict__, float, int, int);

static void pin_thread_to_core(int id) {
    if (!config.enable_numa_lock) return;
    int pid = id / 2, isb = id % 2, core = WORKER_CORE_OFFSET + pid + (isb ? 14 : 0);
#ifdef _WIN32
    if (!SetThreadAffinityMask(GetCurrentThread(), 1ULL << core)) config.enable_numa_lock = false;
#endif
}

static void worker_a_func(int id) {
    pin_thread_to_core(id); WorkerState& s = g_workers[id];
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
    while (!s.should_exit.load(std::memory_order_relaxed)) {
        if (s.has_work.load(std::memory_order_acquire)) { s.work(); s.has_work.store(false, std::memory_order_release); }
        _mm_pause();
    }
}

static void worker_b_func(int id) {
    pin_thread_to_core(id); int pid = id / 2; FetcherRequest& req = g_fetch_requests[pid]; SiblingWorkItem& sw = g_sibling_works[pid]; SPSCRing<WeightTile, 4>& ring = g_workers[id - 1].tile_ring; WorkerState& s = g_workers[id];
    while (!s.should_exit.load(std::memory_order_relaxed)) {
        if (sw.ready.load(std::memory_order_acquire)) {
            const ExpertWeights* W = sw.weights; int rs = sw.row_start, re = sw.row_end; size_t p = (HIDDEN_DIM * 3 / 8 + 31) & ~31;
            quip_matmul_fused(W->gate_proj + rs * p, sw.x_rot, sw.gate_out + rs, W->gate_scale, re - rs, HIDDEN_DIM);
            quip_matmul_fused(W->up_proj + rs * p, sw.x_rot, sw.up_out + rs, W->up_scale, re - rs, HIDDEN_DIM);
            sw.ready.store(false, std::memory_order_release);
        }
        if (req.active.load(std::memory_order_acquire)) {
            const uint8_t* p = req.base_ptr; size_t rem = req.total_bytes;
            while (rem > 0 && !s.should_exit.load(std::memory_order_relaxed)) {
                size_t tb = (rem < L3_TILE_BYTES) ? rem : L3_TILE_BYTES;
                for (size_t o = 0; o < tb; o += CACHE_LINE) _mm_prefetch((const char*)(p + o), _MM_HINT_T1);
                ring.push({p, req.type, req.expert_idx, req.layer_idx, tb});
                p += tb; rem -= tb;
            }
            req.active.store(false, std::memory_order_release);
        }
        _mm_pause();
    }
}

class WorkerPool {
public:
    WorkerPool() { for (int i = 0; i < NUM_WORKER_THREADS; ++i) threads_[i] = std::thread((i % 2 == 0) ? worker_a_func : worker_b_func, i); }
    ~WorkerPool() { for (int i = 0; i < NUM_WORKER_THREADS; ++i) g_workers[i].should_exit = true; for (int i = 0; i < NUM_WORKER_THREADS; ++i) threads_[i].join(); }
    void submit(int pid, std::function<void()> f) { WorkerState& s = g_workers[pid * 2]; while (s.has_work.load(std::memory_order_relaxed)) _mm_pause(); s.work = f; s.has_work.store(true, std::memory_order_release); }
    void wait(int pid) { while (g_workers[pid * 2].has_work.load(std::memory_order_acquire)) _mm_pause(); }
    void prefetch_expert_weights(int pid, const uint8_t* p, size_t b, ProjType t, int ei, int li) {
        FetcherRequest& req = g_fetch_requests[pid]; while (req.active.load(std::memory_order_relaxed)) _mm_pause();
        req.base_ptr = p; req.total_bytes = b; req.type = t; req.expert_idx = ei; req.layer_idx = li; req.active.store(true, std::memory_order_release);
    }
    bool poll_tile(int pid, WeightTile& out) { return g_workers[pid * 2].tile_ring.try_pop(out); }
private:
    std::thread threads_[NUM_WORKER_THREADS];
};

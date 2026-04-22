// =============================================================================
// worker_pool.cpp — CPU worker thread pool.
// =============================================================================

#include "../../include/common.h"
#include <thread>
#include <atomic>
#include <vector>
#include <immintrin.h>
#include <functional>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <pthread.h>
#endif

template<typename T, size_t N>
struct SPSCRing {
    T        buf[N];
    alignas(64) std::atomic<size_t> head{0};
    alignas(64) std::atomic<size_t> tail{0};
    bool push(const T& val) {
        size_t h = head.load(std::memory_order_relaxed);
        if (h - tail.load(std::memory_order_acquire) == N) return false;
        buf[h % N] = val; head.store(h + 1, std::memory_order_release);
        return true;
    }
    bool try_pop(T& out) {
        size_t t = tail.load(std::memory_order_relaxed);
        if (t == head.load(std::memory_order_acquire)) return false;
        out = buf[t % N]; tail.store(t + 1, std::memory_order_release);
        return true;
    }
};

struct WorkerState {
    std::atomic<bool>     has_work{false};
    std::atomic<bool>     should_exit{false};
    std::function<void()> work;
    SPSCRing<WeightTile, 1> tile_ring; 
};

static WorkerState     g_workers[NUM_WORKER_THREADS];
static FetcherRequest  g_fetch_requests[NUM_WORKER_THREADS / 2];
SiblingWorkItem g_sibling_works[NUM_WORKER_THREADS / 2];

static void pin_thread_to_core(int id) {
    int pid = id / 2;
    int isb = id % 2;
    int core = 2 + pid + (isb ? 14 : 0);
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1ULL << core);
#else
    cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

extern void quip_matmul_fused(const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, float row_scale, int rows, int cols);

static void worker_a_func(int id) {
    pin_thread_to_core(id); WorkerState& s = g_workers[id];
    SiblingWorkItem& sw = g_sibling_works[id / 2];
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
    while (!s.should_exit.load(std::memory_order_relaxed)) {
        if (s.has_work.load(std::memory_order_acquire)) { 
            s.work(); 
            s.has_work.store(false, std::memory_order_release); 
        }
        if (sw.ready.load(std::memory_order_acquire)) {
            const ExpertWeights* W = sw.weights; int rs = sw.row_start, re = sw.row_end; 
            size_t pitch = (HIDDEN_DIM * 3 / 8 + 31) & ~31;
            quip_matmul_fused(W->gate_proj + (size_t)rs * pitch, sw.x_rot, sw.gate_out + rs, W->gate_scale, re - rs, HIDDEN_DIM);
            quip_matmul_fused(W->up_proj + (size_t)rs * pitch, sw.x_rot, sw.up_out + rs, W->up_scale, re - rs, HIDDEN_DIM);
            sw.ready.store(false, std::memory_order_release);
        }
        _mm_pause();
    }
}

static void worker_b_func(int id) {
    pin_thread_to_core(id); int pid = id / 2; FetcherRequest& req = g_fetch_requests[pid]; 
    SPSCRing<WeightTile, 1>& ring = g_workers[id - 1].tile_ring; WorkerState& s = g_workers[id];
    while (!s.should_exit.load(std::memory_order_relaxed)) {
        if (req.active.load(std::memory_order_acquire)) {
            const uint8_t* p = req.base_ptr; size_t rem = req.total_bytes;
            while (rem > 0 && !s.should_exit.load(std::memory_order_relaxed)) {
                size_t tb = (rem < L3_TILE_BYTES) ? rem : L3_TILE_BYTES;
                for (size_t i = 0; i < tb; i += 64) _mm_prefetch((const char*)p + i, _MM_HINT_T0);
                while(!ring.push({p, req.type, req.expert_idx, req.layer_idx, tb})) { _mm_pause(); }
                p += tb; rem -= tb;
            }
            req.active.store(false, std::memory_order_release);
        }
        _mm_pause();
    }
}

class WorkerPool {
public:
    WorkerPool() {
        for (int i = 0; i < NUM_WORKER_THREADS; ++i) {
            if (i % 2 == 0) threads_[i] = std::thread(worker_a_func, i);
            else threads_[i] = std::thread(worker_b_func, i);
        }
    }
    ~WorkerPool() {
        for (int i = 0; i < NUM_WORKER_THREADS; ++i) g_workers[i].should_exit.store(true, std::memory_order_relaxed);
        for (int i = 0; i < NUM_WORKER_THREADS; ++i) if (threads_[i].joinable()) threads_[i].join();
    }
    void submit(int pid, std::function<void()> work) {
        g_workers[pid].work = std::move(work); 
        g_workers[pid].has_work.store(true, std::memory_order_release);
    }
    void wait(int pid) { while (g_workers[pid].has_work.load(std::memory_order_acquire)) _mm_pause(); }
    
    void prefetch_expert_weights(int pid, const uint8_t* ptr, size_t bytes, ProjType type, int ei, int li) {
        FetcherRequest& req = g_fetch_requests[pid / 2];
        while (req.active.load(std::memory_order_acquire)) _mm_pause();
        req.base_ptr = ptr; req.total_bytes = bytes; req.type = type; req.expert_idx = ei; req.layer_idx = li;
        req.active.store(true, std::memory_order_release);
    }
    bool poll_tile(int pid, WeightTile& out) { return g_workers[pid].tile_ring.try_pop(out); }
private:
    std::thread threads_[NUM_WORKER_THREADS];
};

// cudagfx_cpu.h: the CPU execution backend (--emit-cpu).
//
// Everything the generated `<name>_cpu.h` needs to run a kernel on the host: the CUDA built-ins, the vector
// arithmetic and swizzles of the CUDA vector types, the math functions, the atomics, and the executor that
// plays the threads of a grid. It is meant for validation, debugging and machines without a usable GPU, not for
// speed: kernels that call __syncthreads() run one OS thread per CUDA thread of a block.
//
// Included explicitly (never by cudagfx_rt.h), so applications that only target the GPU do not pay for it.
//
// Executor of kernels that call __syncthreads():
//   * Linux x86-64: the threads of a block are fibers (user-space stacks, ~10 ns per switch, no system call),
//     run by a round-robin scheduler on the calling thread. Deterministic, and about two orders of magnitude
//     faster than OS threads.
//   * elsewhere (Windows, macOS, AArch64...), and under AddressSanitizer / ThreadSanitizer (which do not follow
//     stack switches): one OS thread per CUDA thread of a block, created once per launch.
//   Define CUDAGFX_CPU_THREADS_ONLY to force the OS-thread executor everywhere.
#pragma once

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "cudagfx_rt.h"

#if defined(__linux__) && defined(__x86_64__) && !defined(CUDAGFX_CPU_THREADS_ONLY) && !defined(__SANITIZE_ADDRESS__) && \
    !defined(__SANITIZE_THREAD__)
#define CUDAGFX_CPU_FIBERS 1
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#undef CUDAGFX_CPU_FIBERS
#endif
#endif
#endif
#ifndef CUDAGFX_CPU_FIBERS
#define CUDAGFX_CPU_FIBERS 0
#endif

namespace cudagfx {
namespace cpu {

// ============================================================================
// Vector types: traits, component access
// ============================================================================

template <typename V> struct VecTraits;
#define CUDAGFX_VEC_TRAITS(V, S, N) \
    template <> struct VecTraits<V> { using Scalar = S; static constexpr int count = N; };
CUDAGFX_VEC_TRAITS(float2, float, 2)
CUDAGFX_VEC_TRAITS(float3, float, 3)
CUDAGFX_VEC_TRAITS(float4, float, 4)
CUDAGFX_VEC_TRAITS(int2, int, 2)
CUDAGFX_VEC_TRAITS(int3, int, 3)
CUDAGFX_VEC_TRAITS(int4, int, 4)
CUDAGFX_VEC_TRAITS(uint2, unsigned int, 2)
CUDAGFX_VEC_TRAITS(uint3, unsigned int, 3)
CUDAGFX_VEC_TRAITS(uint4, unsigned int, 4)
#undef CUDAGFX_VEC_TRAITS

// The vector type with `N` components of scalar type `S`.
template <typename S, int N> struct VecOf;
template <> struct VecOf<float, 2> { using type = float2; };
template <> struct VecOf<float, 3> { using type = float3; };
template <> struct VecOf<float, 4> { using type = float4; };
template <> struct VecOf<int, 2> { using type = int2; };
template <> struct VecOf<int, 3> { using type = int3; };
template <> struct VecOf<int, 4> { using type = int4; };
template <> struct VecOf<unsigned int, 2> { using type = uint2; };
template <> struct VecOf<unsigned int, 3> { using type = uint3; };
template <> struct VecOf<unsigned int, 4> { using type = uint4; };

// Components of a vector are contiguous scalars of the same type (static_assert'ed in the compat header).
template <typename V>
inline typename VecTraits<V>::Scalar& comp(V& v, int i) {
    return reinterpret_cast<typename VecTraits<V>::Scalar*>(&v)[i];
}
template <typename V>
inline const typename VecTraits<V>::Scalar& comp(const V& v, int i) {
    return reinterpret_cast<const typename VecTraits<V>::Scalar*>(&v)[i];
}

constexpr int swizzleIndex(char c) { return c == 'x' ? 0 : c == 'y' ? 1 : c == 'z' ? 2 : 3; }

// v.xy, v.zyx...: a copy holding the selected components (at least 2).
template <char... C, typename V>
inline typename VecOf<typename VecTraits<V>::Scalar, static_cast<int>(sizeof...(C))>::type swz(const V& v) {
    static_assert(sizeof...(C) >= 2 && sizeof...(C) <= 4, "a swizzle selects 2 to 4 components");
    typename VecOf<typename VecTraits<V>::Scalar, static_cast<int>(sizeof...(C))>::type r{};
    const int idx[] = {swizzleIndex(C)...};
    for (int k = 0; k < static_cast<int>(sizeof...(C)); ++k) comp(r, k) = comp(v, idx[k]);
    return r;
}

// v.xy = ..., v.zw *= ...: a writable view onto the selected components.
template <typename V, char... C>
struct SwizzleRef {
    using Scalar = typename VecTraits<V>::Scalar;
    using R = typename VecOf<Scalar, static_cast<int>(sizeof...(C))>::type;
    V& v;
    SwizzleRef& operator=(const R& r) {
        const int idx[] = {swizzleIndex(C)...};
        for (int k = 0; k < static_cast<int>(sizeof...(C)); ++k) comp(v, idx[k]) = comp(r, k);
        return *this;
    }
    operator R() const { return swz<C...>(v); }
    SwizzleRef& operator+=(const R& r) { return *this = static_cast<R>(*this) + r; }
    SwizzleRef& operator-=(const R& r) { return *this = static_cast<R>(*this) - r; }
    SwizzleRef& operator*=(const R& r) { return *this = static_cast<R>(*this) * r; }
    SwizzleRef& operator/=(const R& r) { return *this = static_cast<R>(*this) / r; }
    SwizzleRef& operator*=(Scalar s) { return *this = static_cast<R>(*this) * s; }
    SwizzleRef& operator/=(Scalar s) { return *this = static_cast<R>(*this) / s; }
    SwizzleRef& operator+=(Scalar s) { return *this = static_cast<R>(*this) + s; }
    SwizzleRef& operator-=(Scalar s) { return *this = static_cast<R>(*this) - s; }
};
template <char... C, typename V>
inline SwizzleRef<V, C...> swizzleRef(V& v) {
    return SwizzleRef<V, C...>{v};
}

inline uint3 toUint3(const dim3& d) { return uint3{d.x, d.y, d.z}; }

} // namespace cpu
} // namespace cudagfx

// ============================================================================
// Component-wise arithmetic on the CUDA vector types (found by argument-dependent lookup: the types live in
// the global namespace). Only the operators CudaGFX accepts in a kernel are provided: + - * / and unary minus.
// ============================================================================

#define CUDAGFX_VEC_BINARY(OP)                                                                                      \
    template <typename V, typename = typename cudagfx::cpu::VecTraits<V>::Scalar>                                   \
    inline V operator OP(const V& a, const V& b) {                                                                  \
        V r{};                                                                                                      \
        for (int i = 0; i < cudagfx::cpu::VecTraits<V>::count; ++i) cudagfx::cpu::comp(r, i) = cudagfx::cpu::comp(a, i) OP cudagfx::cpu::comp(b, i); \
        return r;                                                                                                   \
    }                                                                                                               \
    template <typename V, typename = typename cudagfx::cpu::VecTraits<V>::Scalar>                                   \
    inline V operator OP(const V& a, typename cudagfx::cpu::VecTraits<V>::Scalar s) {                               \
        V r{};                                                                                                      \
        for (int i = 0; i < cudagfx::cpu::VecTraits<V>::count; ++i) cudagfx::cpu::comp(r, i) = cudagfx::cpu::comp(a, i) OP s; \
        return r;                                                                                                   \
    }                                                                                                               \
    template <typename V, typename = typename cudagfx::cpu::VecTraits<V>::Scalar>                                   \
    inline V operator OP(typename cudagfx::cpu::VecTraits<V>::Scalar s, const V& b) {                               \
        V r{};                                                                                                      \
        for (int i = 0; i < cudagfx::cpu::VecTraits<V>::count; ++i) cudagfx::cpu::comp(r, i) = s OP cudagfx::cpu::comp(b, i); \
        return r;                                                                                                   \
    }
CUDAGFX_VEC_BINARY(+)
CUDAGFX_VEC_BINARY(-)
CUDAGFX_VEC_BINARY(*)
CUDAGFX_VEC_BINARY(/)
#undef CUDAGFX_VEC_BINARY

#define CUDAGFX_VEC_COMPOUND(OP)                                                                                    \
    template <typename V, typename = typename cudagfx::cpu::VecTraits<V>::Scalar>                                   \
    inline V& operator OP##=(V& a, const V& b) {                                                                    \
        a = a OP b;                                                                                                 \
        return a;                                                                                                   \
    }                                                                                                               \
    template <typename V, typename = typename cudagfx::cpu::VecTraits<V>::Scalar>                                   \
    inline V& operator OP##=(V& a, typename cudagfx::cpu::VecTraits<V>::Scalar s) {                                 \
        a = a OP s;                                                                                                 \
        return a;                                                                                                   \
    }
CUDAGFX_VEC_COMPOUND(+)
CUDAGFX_VEC_COMPOUND(-)
CUDAGFX_VEC_COMPOUND(*)
CUDAGFX_VEC_COMPOUND(/)
#undef CUDAGFX_VEC_COMPOUND

template <typename V, typename = typename cudagfx::cpu::VecTraits<V>::Scalar>
inline V operator-(const V& a) {
    V r{};
    for (int i = 0; i < cudagfx::cpu::VecTraits<V>::count; ++i) cudagfx::cpu::comp(r, i) = -cudagfx::cpu::comp(a, i);
    return r;
}

namespace cudagfx {
namespace cpu {

// ============================================================================
// Built-ins and math (names as in CUDA, so that the translated kernel bodies read the same)
// ============================================================================

inline thread_local dim3 threadIdx;
inline thread_local dim3 blockIdx;
inline thread_local dim3 blockDim;
inline thread_local dim3 gridDim;

using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::ceil;
using std::cos;
using std::cosh;
using std::exp;
using std::exp2;
using std::fabs;
using std::floor;
using std::fmax;
using std::fmin;
using std::fmod;
using std::isinf;
using std::isnan;
using std::log;
using std::log2;
using std::max;
using std::min;
using std::pow;
using std::round;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;
using std::trunc;

// C23 (glibc >= 2.41) declares rsqrt / rsqrtf in <math.h>. A function template cannot make the call ambiguous with them:
// for a float argument the library's function is preferred, and the template still serves the libraries without it.
template <typename T>
inline float rsqrtf(T x) {
    return 1.0f / std::sqrt(static_cast<float>(x));
}
template <typename T>
inline float rsqrt(T x) {
    return 1.0f / std::sqrt(static_cast<float>(x));
}
inline float __sinf(float x) { return std::sin(x); }
inline float __cosf(float x) { return std::cos(x); }
inline float __tanf(float x) { return std::tan(x); }
inline float __expf(float x) { return std::exp(x); }
inline float __logf(float x) { return std::log(x); }
inline float __log2f(float x) { return std::log2(x); }
inline float __exp2f(float x) { return std::exp2(x); }
inline float __powf(float x, float y) { return std::pow(x, y); }

inline int __float_as_int(float v) { int r; std::memcpy(&r, &v, sizeof r); return r; }
inline unsigned int __float_as_uint(float v) { unsigned int r; std::memcpy(&r, &v, sizeof r); return r; }
inline float __int_as_float(int v) { float r; std::memcpy(&r, &v, sizeof r); return r; }
inline float __uint_as_float(unsigned int v) { float r; std::memcpy(&r, &v, sizeof r); return r; }
inline int __popc(unsigned int v) {
    int n = 0;
    for (; v != 0u; v &= v - 1u) ++n;
    return n;
}

// ============================================================================
// Atomics (each returns the previous value, like CUDA)
// ============================================================================

inline int atomicAdd(int* a, int v) { return __atomic_fetch_add(a, v, __ATOMIC_SEQ_CST); }
inline unsigned int atomicAdd(unsigned int* a, unsigned int v) { return __atomic_fetch_add(a, v, __ATOMIC_SEQ_CST); }
inline int atomicSub(int* a, int v) { return __atomic_fetch_sub(a, v, __ATOMIC_SEQ_CST); }
inline unsigned int atomicSub(unsigned int* a, unsigned int v) { return __atomic_fetch_sub(a, v, __ATOMIC_SEQ_CST); }
inline int atomicAnd(int* a, int v) { return __atomic_fetch_and(a, v, __ATOMIC_SEQ_CST); }
inline unsigned int atomicAnd(unsigned int* a, unsigned int v) { return __atomic_fetch_and(a, v, __ATOMIC_SEQ_CST); }
inline int atomicOr(int* a, int v) { return __atomic_fetch_or(a, v, __ATOMIC_SEQ_CST); }
inline unsigned int atomicOr(unsigned int* a, unsigned int v) { return __atomic_fetch_or(a, v, __ATOMIC_SEQ_CST); }
inline int atomicXor(int* a, int v) { return __atomic_fetch_xor(a, v, __ATOMIC_SEQ_CST); }
inline unsigned int atomicXor(unsigned int* a, unsigned int v) { return __atomic_fetch_xor(a, v, __ATOMIC_SEQ_CST); }
inline int atomicExch(int* a, int v) { return __atomic_exchange_n(a, v, __ATOMIC_SEQ_CST); }
inline unsigned int atomicExch(unsigned int* a, unsigned int v) { return __atomic_exchange_n(a, v, __ATOMIC_SEQ_CST); }

namespace detail {
template <typename T, typename Better>
inline T atomicExtremum(T* a, T v, Better better) {
    T old = __atomic_load_n(a, __ATOMIC_SEQ_CST);
    while (better(v, old) && !__atomic_compare_exchange_n(a, &old, v, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
    }
    return old;
}
} // namespace detail

inline int atomicMin(int* a, int v) { return detail::atomicExtremum(a, v, [](int x, int y) { return x < y; }); }
inline unsigned int atomicMin(unsigned int* a, unsigned int v) { return detail::atomicExtremum(a, v, [](unsigned x, unsigned y) { return x < y; }); }
inline int atomicMax(int* a, int v) { return detail::atomicExtremum(a, v, [](int x, int y) { return x > y; }); }
inline unsigned int atomicMax(unsigned int* a, unsigned int v) { return detail::atomicExtremum(a, v, [](unsigned x, unsigned y) { return x > y; }); }

inline int atomicCAS(int* a, int compare, int v) {
    __atomic_compare_exchange_n(a, &compare, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return compare;
}
inline unsigned int atomicCAS(unsigned int* a, unsigned int compare, unsigned int v) {
    __atomic_compare_exchange_n(a, &compare, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return compare;
}

// float atomics go through the 32-bit pattern (memcpy: no aliasing violation).
inline float atomicAdd(float* a, float v) {
    unsigned int* bits = reinterpret_cast<unsigned int*>(a);
    unsigned int oldBits = __atomic_load_n(bits, __ATOMIC_SEQ_CST);
    for (;;) {
        float oldVal;
        std::memcpy(&oldVal, &oldBits, sizeof oldVal);
        const float sum = oldVal + v;
        unsigned int newBits;
        std::memcpy(&newBits, &sum, sizeof newBits);
        if (__atomic_compare_exchange_n(bits, &oldBits, newBits, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) return oldVal;
    }
}
inline float atomicExch(float* a, float v) {
    unsigned int newBits;
    std::memcpy(&newBits, &v, sizeof newBits);
    const unsigned int oldBits = __atomic_exchange_n(reinterpret_cast<unsigned int*>(a), newBits, __ATOMIC_SEQ_CST);
    float old;
    std::memcpy(&old, &oldBits, sizeof old);
    return old;
}

// ============================================================================
// Block synchronisation and kernel execution
// ============================================================================

// Barrier of one thread block (__syncthreads()). A thread that leaves the kernel `drop()`s out: it no longer
// counts, exactly as on a GPU where exited threads do not hold up a barrier. (A fixed-size barrier would
// deadlock on the very common `if (i >= n) return;` placed before a __syncthreads().)
class BlockBarrier {
public:
    explicit BlockBarrier(unsigned participants = 1) : expected_(participants) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        const unsigned long long gen = generation_;
        if (++arrived_ >= expected_) {
            release();
        } else {
            cv_.wait(lock, [this, gen] { return gen != generation_; });
        }
    }

    void drop() {
        std::unique_lock<std::mutex> lock(mtx_);
        if (expected_ > 0) --expected_;
        if (expected_ > 0 && arrived_ >= expected_) release();
    }

    // Only when no thread uses the barrier (between two blocks).
    void reset(unsigned participants) {
        std::unique_lock<std::mutex> lock(mtx_);
        expected_ = participants;
        arrived_ = 0;
    }

private:
    void release() {
        arrived_ = 0;
        ++generation_;
        cv_.notify_all();
    }

    std::mutex mtx_;
    std::condition_variable cv_;
    unsigned expected_;
    unsigned arrived_ = 0;
    unsigned long long generation_ = 0;
};

inline thread_local BlockBarrier* currentBarrier = nullptr;

#if CUDAGFX_CPU_FIBERS
namespace detail {

// Minimal context switch (System V x86-64): the callee-saved registers are pushed on the stack of the fiber that
// is suspended, whose stack pointer is stored in *saveSp; then the stack of the fiber to resume is installed
// and its registers popped. `ret` continues where that fiber had called fiberSwitch (or at fiberEntry, for a
// fiber that has not started yet).
__attribute__((naked, noinline, unused)) static void fiberSwitch([[maybe_unused]] void** saveSp, [[maybe_unused]] void* newSp) {
    __asm__ volatile(
        "pushq %rbp\n\t"
        "pushq %rbx\n\t"
        "pushq %r12\n\t"
        "pushq %r13\n\t"
        "pushq %r14\n\t"
        "pushq %r15\n\t"
        "movq %rsp, (%rdi)\n\t"
        "movq %rsi, %rsp\n\t"
        "popq %r15\n\t"
        "popq %r14\n\t"
        "popq %r13\n\t"
        "popq %r12\n\t"
        "popq %rbx\n\t"
        "popq %rbp\n\t"
        "ret\n\t");
}

struct FiberTask {
    void* sp = nullptr;                // saved stack pointer while the fiber is suspended
    void** schedulerSp = nullptr;      // where the scheduler's own stack pointer is kept
    void (*body)(void*) = nullptr;
    void* arg = nullptr;
    bool done = false;
};

inline thread_local FiberTask* currentTask = nullptr;  // the fiber running on this OS thread (null outside a kernel)
inline thread_local FiberTask* startingTask = nullptr; // hands the task to fiberEntry on a fiber's first run

[[maybe_unused]] static void fiberEntry() {
    FiberTask* task = startingTask;
    task->body(task->arg);
    task->done = true;
    fiberSwitch(&task->sp, *task->schedulerSp); // a finished fiber is never resumed
    __builtin_trap();
}

// Lays out the first frame of a fiber so that the first switch "returns" into fiberEntry, with the alignment the
// ABI expects at a function entry (rsp % 16 == 8).
inline void prepareFiber(FiberTask& task, unsigned char* stack, std::size_t size) {
    std::uintptr_t top = reinterpret_cast<std::uintptr_t>(stack + size) & ~static_cast<std::uintptr_t>(15);
    void** sp = reinterpret_cast<void**>(top);
    *--sp = nullptr;                                  // fake return address of fiberEntry (never used)
    *--sp = reinterpret_cast<void*>(&fiberEntry);     // popped by the `ret` of the first switch
    for (int i = 0; i < 6; ++i) *--sp = nullptr;      // rbp rbx r12 r13 r14 r15
    task.sp = sp;
    task.done = false;
}

constexpr std::size_t kFiberStackBytes = 128 * 1024;

} // namespace detail
#endif

// __syncthreads(): on a fiber, hand control back to the scheduler (which resumes this fiber once every other fiber
// of the block has reached its barrier or left the kernel); on an OS thread, wait on the block barrier.
inline void syncthreads() {
#if CUDAGFX_CPU_FIBERS
    if (detail::currentTask) {
        detail::fiberSwitch(&detail::currentTask->sp, *detail::currentTask->schedulerSp);
        return;
    }
#endif
    if (currentBarrier) currentBarrier->wait();
}

namespace detail {
inline dim3 unflatten(unsigned index, dim3 extent) {
    return dim3(index % extent.x, (index / extent.x) % extent.y, index / (extent.x * extent.y));
}

#if CUDAGFX_CPU_FIBERS
// The threads of a block as fibers. One scheduling round resumes every unfinished fiber once: each runs until
// its next __syncthreads() (it suspends) or until it leaves the kernel (it finishes). When a round ends, all
// the fibers still alive are at a barrier, which is therefore released by the next round. A thread that has
// left the kernel simply stops being scheduled, so it never holds up a barrier (same as on a GPU).
template <typename Shared, typename ThreadFn>
inline void runFibers(dim3 grid, dim3 blk, unsigned threadsPerBlock, unsigned blocks, ThreadFn& thread) {
    struct Context {
        ThreadFn* fn;
        Shared* shared;
    };
    Shared shared{};
    Context ctx{&thread, &shared};
    void* schedulerSp = nullptr;

    std::vector<FiberTask> tasks(threadsPerBlock);
    std::unique_ptr<unsigned char[]> stacks(new unsigned char[static_cast<std::size_t>(threadsPerBlock) * kFiberStackBytes]);
    for (unsigned t = 0; t < threadsPerBlock; ++t) {
        tasks[t].schedulerSp = &schedulerSp;
        tasks[t].arg = &ctx;
        tasks[t].body = [](void* a) {
            Context* c = static_cast<Context*>(a);
            (*c->fn)(*c->shared);
        };
    }

    blockDim = blk;
    gridDim = grid;
    for (unsigned b = 0; b < blocks; ++b) {
        blockIdx = unflatten(b, grid);
        shared = Shared{};
        for (unsigned t = 0; t < threadsPerBlock; ++t) {
            prepareFiber(tasks[t], stacks.get() + static_cast<std::size_t>(t) * kFiberStackBytes, kFiberStackBytes);
        }
        unsigned alive = threadsPerBlock;
        while (alive > 0) {
            for (unsigned t = 0; t < threadsPerBlock; ++t) {
                FiberTask& task = tasks[t];
                if (task.done) continue;
                threadIdx = unflatten(t, blk);
                currentTask = &task;
                startingTask = &task; // only read on the first run of the fiber
                fiberSwitch(&schedulerSp, task.sp);
                currentTask = nullptr;
                if (task.done) --alive;
            }
        }
    }
}
#endif

template <typename Shared, typename ThreadFn>
inline void runThreads(dim3 grid, dim3 blk, unsigned threadsPerBlock, unsigned blocks, ThreadFn& thread) {
    Shared shared{};
    BlockBarrier kernelBarrier(threadsPerBlock); // __syncthreads() inside the kernel
    BlockBarrier phaseBarrier(threadsPerBlock);  // start / end of each block (never dropped)
    std::vector<std::thread> workers;
    workers.reserve(threadsPerBlock);
    for (unsigned t = 0; t < threadsPerBlock; ++t) {
        workers.emplace_back([&, t]() {
            threadIdx = unflatten(t, blk);
            blockDim = blk;
            gridDim = grid;
            currentBarrier = &kernelBarrier;
            for (unsigned b = 0; b < blocks; ++b) {
                if (t == 0) {
                    shared = Shared{};
                    kernelBarrier.reset(threadsPerBlock);
                }
                phaseBarrier.wait(); // everybody sees the fresh shared memory and barrier
                blockIdx = unflatten(b, grid);
                thread(shared);
                kernelBarrier.drop(); // this thread has left the kernel
                phaseBarrier.wait();  // the whole block is done before the next one starts
            }
            currentBarrier = nullptr;
        });
    }
    for (auto& w : workers) w.join();
}
} // namespace detail

// Plays a whole grid. `Shared` is the kernel's __shared__ state (value-initialised for each block) and
// `thread(shared)` runs the body for the current threadIdx/blockIdx. Blocks run one after the other.
template <typename Shared, typename ThreadFn>
inline void runKernel(dim3 grid, dim3 blk, bool usesBarrier, ThreadFn&& thread) {
    const unsigned threadsPerBlock = blk.x * blk.y * blk.z;
    const unsigned blocks = grid.x * grid.y * grid.z;
    if (threadsPerBlock == 0 || blocks == 0) return;

    if (!usesBarrier || threadsPerBlock == 1) {
        blockDim = blk;
        gridDim = grid;
        for (unsigned b = 0; b < blocks; ++b) {
            blockIdx = detail::unflatten(b, grid);
            Shared shared{};
            for (unsigned t = 0; t < threadsPerBlock; ++t) {
                threadIdx = detail::unflatten(t, blk);
                thread(shared);
            }
        }
        return;
    }
#if CUDAGFX_CPU_FIBERS
    detail::runFibers<Shared>(grid, blk, threadsPerBlock, blocks, thread);
#else
    detail::runThreads<Shared>(grid, blk, threadsPerBlock, blocks, thread);
#endif
}

} // namespace cpu
} // namespace cudagfx

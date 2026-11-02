// Test of the runtime WITHOUT a GPU (CUDAGFX_NO_BGFX): memory, copies, bounds, launch validation, error codes.
// Compiled with -fno-exceptions -fno-rtti to check the runtime constraint C12.
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "CudaGFX/cudagfx_cpu.h"
#include "CudaGFX/cudagfx_rt.h"

namespace {
int g_failures = 0;
int g_checks = 0;
}
#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_checks;                                                                   \
        if (!(cond)) {                                                                \
            std::printf("FAILED %s:%d : %s\n", __FILE__, __LINE__, #cond);             \
            ++g_failures;                                                             \
        }                                                                             \
    } while (0)

using cudagfx::Access;
using cudagfx::ErrorCode;
using cudagfx::Launch;
using cudagfx::ShaderBinaryTable;

static ShaderBinaryTable table() {
    ShaderBinaryTable t;
    t.kernel = "test";
    t.argsUniform = "u_test_args";
    t.gridUniform = "u_test_grid";
    t.block[0] = 64;
    return t;
}

static void clearError() { (void)cudaGetLastError(); }

static void testAllocation() {
    void* a = nullptr;
    void* b = nullptr;
    CHECK(cudaMalloc(&a, 1000) == cudaSuccess && a != nullptr);
    CHECK(cudaMalloc(&b, 1000) == cudaSuccess && b != nullptr);
    CHECK(a != b);
    // base + size belongs to no neighbouring allocation (guard zone)
    char* pastA = static_cast<char*>(a) + 1000;
    CHECK(pastA != b);
    // null allocation: success and null pointer, as in CUDA
    void* z = reinterpret_cast<void*>(1);
    CHECK(cudaMalloc(&z, 0) == cudaSuccess && z == nullptr);
    CHECK(cudaMalloc(nullptr, 8) == cudaErrorInvalidValue);
    clearError();
    CHECK(cudaFree(nullptr) == cudaSuccess);
    CHECK(cudaFree(static_cast<char*>(a) + 8) == cudaErrorInvalidValue); // not the start of the allocation
    CHECK(cudagfx::lastErrorCode() == ErrorCode::BadPointer);
    clearError();
    CHECK(cudaFree(a) == cudaSuccess);
    CHECK(cudaFree(a) == cudaErrorInvalidValue); // double free
    clearError();
    CHECK(cudaFree(b) == cudaSuccess);
    void* huge = nullptr;
    CHECK(cudaMalloc(&huge, static_cast<size_t>(1) << 40) == cudaErrorMemoryAllocation);
    clearError();
}

static void testCopies() {
    std::vector<int> host(64), back(64, -1);
    for (int i = 0; i < 64; ++i) host[i] = i * 3;
    int* d = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d), 64 * sizeof(int)) == cudaSuccess);
    CHECK(cudaMemcpy(d, host.data(), 64 * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(back.data(), d, 64 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(back == host);

    // partial copy at an offset
    int patch[4] = {-7, -8, -9, -10};
    CHECK(cudaMemcpy(d + 10, patch, sizeof(patch), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(back.data(), d, 64 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(back[9] == 27 && back[10] == -7 && back[13] == -10 && back[14] == 42);

    // out of bounds: refused BEFORE any write (no corruption)
    struct { int before[4]; int canary[4]; } guard = {{1, 2, 3, 4}, {0x5A5A, 0x5A5A, 0x5A5A, 0x5A5A}};
    CHECK(cudaMemcpy(d + 60, host.data(), 8 * sizeof(int), cudaMemcpyHostToDevice) == cudaErrorInvalidValue);
    CHECK(cudagfx::lastErrorCode() == ErrorCode::OutOfBounds);
    clearError();
    CHECK(cudaMemcpy(guard.before, d + 62, 8 * sizeof(int), cudaMemcpyDeviceToHost) == cudaErrorInvalidValue);
    CHECK(guard.canary[0] == 0x5A5A && guard.canary[3] == 0x5A5A);
    clearError();

    // invalid pointers
    CHECK(cudaMemcpy(back.data(), host.data(), 4, cudaMemcpyHostToDevice) == cudaErrorInvalidValue); // host destination
    clearError();
    CHECK(cudaMemcpy(nullptr, host.data(), 4, cudaMemcpyHostToHost) == cudaErrorInvalidValue);
    clearError();
    CHECK(cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyHostToHost) == cudaSuccess); // 0 bytes: always valid

    // cudaMemcpyDefault infers the direction from the pointers
    std::vector<int> other(64, 0);
    CHECK(cudaMemcpy(d, other.data(), 64 * sizeof(int), cudaMemcpyDefault) == cudaSuccess);
    CHECK(cudaMemcpy(back.data(), d, 64 * sizeof(int), cudaMemcpyDefault) == cudaSuccess);
    CHECK(back == other);
    CHECK(cudaMemcpy(other.data(), host.data(), 16, cudaMemcpyDefault) == cudaSuccess && other[1] == 3);

    // device -> device: same allocation with overlap, then two allocations
    CHECK(cudaMemcpy(d, host.data(), 64 * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d + 2, d, 8 * sizeof(int), cudaMemcpyDeviceToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(back.data(), d, 12 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(back[0] == 0 && back[2] == 0 && back[3] == 3 && back[9] == 21 && back[10] == 30); // memmove: d[2..9] = old d[0..7]
    int* d2 = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d2), 64 * sizeof(int)) == cudaSuccess);
    CHECK(cudaMemcpy(d2, d, 64 * sizeof(int), cudaMemcpyDeviceToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d2 + 60, d, 8 * sizeof(int), cudaMemcpyDeviceToDevice) == cudaErrorInvalidValue);
    clearError();

    // cudaMemset
    std::vector<int> before(64);
    CHECK(cudaMemcpy(before.data(), d2, 64 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemset(d2, 0x11, 16) == cudaSuccess);
    CHECK(cudaMemcpy(back.data(), d2, 64 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(back[0] == 0x11111111 && back[3] == 0x11111111 && back[4] == before[4] && back[63] == before[63]);
    CHECK(cudaMemset(d2, 0, 64 * sizeof(int) + 1) == cudaErrorInvalidValue);
    clearError();
    CHECK(cudaMemset(host.data(), 0, 4) == cudaErrorInvalidValue);
    clearError();

    CHECK(cudaFree(d) == cudaSuccess && cudaFree(d2) == cudaSuccess);
}

static void testErrorState() {
    clearError();
    CHECK(cudaPeekAtLastError() == cudaSuccess);
    void* p = nullptr;
    CHECK(cudaMalloc(nullptr, 1) != cudaSuccess);
    CHECK(cudaPeekAtLastError() == cudaErrorInvalidValue);
    CHECK(cudaPeekAtLastError() == cudaErrorInvalidValue); // Peek does not consume the error
    CHECK(std::strlen(cudagfx::lastErrorMessage()) > 0);
    CHECK(cudaGetLastError() == cudaErrorInvalidValue);
    CHECK(cudaGetLastError() == cudaSuccess);              // consumed
    CHECK(cudagfx::lastErrorCode() == ErrorCode::None);
    CHECK(std::strcmp(cudaGetErrorString(cudaErrorInvalidConfiguration), "cudaErrorInvalidConfiguration") == 0);
    CHECK(std::strcmp(cudagfx::errorCodeName(ErrorCode::PointerAlias), "CGFXR003") == 0);
    (void)p;
}

static void testLaunchValidation() {
    float* f = nullptr;
    float* g = nullptr;
    int* i = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&f), 1024 * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(reinterpret_cast<void**>(&g), 1024 * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(reinterpret_cast<void**>(&i), 1024 * sizeof(int)) == cudaSuccess);

    {   // unknown pointer, null
        Launch l(table());
        float local[4] = {};
        CHECK(!l.buffer<float>(0, local, Access::Read));
        CHECK(cudagfx::lastErrorCode() == ErrorCode::BadPointer);
        clearError();
        Launch l2(table());
        CHECK(!l2.buffer<float>(0, static_cast<const float*>(nullptr), Access::Read));
        CHECK(cudagfx::lastErrorCode() == ErrorCode::BadPointer);
        clearError();
    }
    {   // offset not aligned to 256 bytes (bgfx::setBuffer) or to the element
        Launch l(table());
        CHECK(!l.buffer<float>(0, f + 1, Access::Read));
        CHECK(cudagfx::lastErrorCode() == ErrorCode::BadPointer);
        clearError();
        Launch l2(table());
        CHECK(!l2.buffer<float>(0, reinterpret_cast<float*>(reinterpret_cast<char*>(f) + 2), Access::Read));
        clearError();
        Launch l3(table());
        CHECK(l3.buffer<float>(0, f + 64, Access::Read)); // 256 bytes: accepted
    }
    {   // alias: two writes on the same buffer, or one read + one write
        Launch l(table());
        CHECK(l.buffer<float>(0, f, Access::ReadWrite));
        CHECK(!l.buffer<float>(1, f, Access::Read));
        CHECK(cudagfx::lastErrorCode() == ErrorCode::PointerAlias);
        clearError();
        Launch ro(table());
        CHECK(ro.buffer<float>(0, g, Access::Read));
        CHECK(ro.buffer<float>(1, g, Access::Read)); // read-only on both sides: harmless
    }
    {   // reinterpretation of an already typed buffer
        Launch l(table());
        CHECK(!l.buffer<int>(0, reinterpret_cast<int*>(f), Access::Read));
        CHECK(cudagfx::lastErrorCode() == ErrorCode::TypeReinterpret);
        clearError();
    }
    {   // size not a multiple of the element size
        void* odd = nullptr;
        CHECK(cudaMalloc(&odd, 10) == cudaSuccess);
        Launch l(table());
        CHECK(!l.buffer<float>(0, static_cast<float*>(odd), Access::Read));
        CHECK(cudagfx::lastErrorCode() == ErrorCode::ElementSize);
        clearError();
        CHECK(cudaFree(odd) == cudaSuccess);
    }
    {   // a failure makes the launch inert: dispatch executes nothing and does not clear the error
        Launch l(table());
        l.buffer<float>(0, f + 1, Access::Read);
        const cudaError_t before = cudaPeekAtLastError();
        l.dispatch(dim3(1), dim3(64));
        CHECK(cudaPeekAtLastError() == before);
        clearError();
    }
    {   // inconsistent block / grid
        Launch l(table());
        l.dispatch(dim3(1), dim3(32));
        CHECK(cudagfx::lastErrorCode() == ErrorCode::BadBlock);
        CHECK(cudaGetLastError() == cudaErrorInvalidConfiguration);
        Launch l2(table());
        l2.dispatch(dim3(0), dim3(64));
        CHECK(cudagfx::lastErrorCode() == ErrorCode::BadBlock);
        clearError();
        cudagfx::detail::badBlockSize("k", dim3(7, 1, 1));
        CHECK(cudagfx::lastErrorCode() == ErrorCode::BadBlock);
        clearError();
    }
    {   // too many by-value arguments: 17 vec4 > 16
        Launch l(table());
        bool ok = true;
        for (int k = 0; k < 17; ++k) ok = l.arg(make_float4(1.f, 2.f, 3.f, 4.f)) && ok;
        CHECK(!ok);
        CHECK(cudagfx::lastErrorCode() == ErrorCode::ArgOverflow);
        clearError();
    }
    {   // every by-value argument type is accepted
        Launch l(table());
        CHECK(l.arg(1.f) && l.arg(2) && l.arg(3u) && l.arg(true));
        CHECK(l.arg(make_float2(1.f, 2.f)) && l.arg(make_float3(1.f, 2.f, 3.f)) && l.arg(make_float4(1.f, 2.f, 3.f, 4.f)));
        CHECK(l.arg(make_int2(1, 2)) && l.arg(make_int3(1, 2, 3)) && l.arg(make_int4(1, 2, 3, 4)));
        CHECK(l.arg(make_uint2(1, 2)) && l.arg(make_uint3(1, 2, 3)) && l.arg(make_uint4(1, 2, 3, 4)));
    }
    {   // without bgfx, a valid launch fails EXPLICITLY (never a silent success)
        Launch l(table());
        CHECK(l.buffer<float>(0, g, Access::ReadWrite));
        l.dispatch(dim3(1), dim3(64));
#if !CUDAGFX_HAS_BGFX
        CHECK(cudagfx::lastErrorCode() == ErrorCode::MissingShader);
        CHECK(cudaGetLastError() == cudaErrorInitializationError);
#endif
        clearError();
    }
    // memory remains usable after a rejected launch
    std::vector<float> host(1024, 2.f), back(1024, 0.f);
    CHECK(cudaMemcpy(g, host.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(back.data(), g, 1024 * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(back == host);

    CHECK(cudaFree(f) == cudaSuccess && cudaFree(g) == cudaSuccess && cudaFree(i) == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
}

struct TestParticle {
    float4 pos;
    float4 vel;
};

static void testPhase6Features() {
    // 1. ExecutionMode
    CHECK(cudagfx::executionMode() == cudagfx::ExecutionMode::GPU);
    cudagfx::setExecutionMode(cudagfx::ExecutionMode::CPU);
    CHECK(cudagfx::executionMode() == cudagfx::ExecutionMode::CPU);
    cudagfx::setExecutionMode(cudagfx::ExecutionMode::GPU);
    CHECK(cudagfx::executionMode() == cudagfx::ExecutionMode::GPU);

    // 2. cudaMemcpyToSymbol / cudaMemcpyFromSymbol
    float symbolData[4] = {1.f, 2.f, 3.f, 4.f};
    float hostSrc[4] = {10.f, 20.f, 30.f, 40.f};
    float hostDst[4] = {0.f, 0.f, 0.f, 0.f};
    CHECK(cudaMemcpyToSymbol(symbolData, hostSrc, sizeof(hostSrc), 0, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(symbolData[0] == 10.f && symbolData[3] == 40.f);
    CHECK(cudaMemcpyFromSymbol(hostDst, symbolData, sizeof(hostDst), 0, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(hostDst[0] == 10.f && hostDst[3] == 40.f);

    // 3. BlockBarrier (a single participant never blocks)
    cudagfx::cpu::BlockBarrier barrier(1);
    barrier.wait();

    // 4. Atomics (each returns the previous value)
    namespace cpu = cudagfx::cpu;
    int val = 10;
    CHECK(cpu::atomicAdd(&val, 5) == 10);
    CHECK(val == 15);
    CHECK(cpu::atomicSub(&val, 3) == 15);
    CHECK(val == 12);
    CHECK(cpu::atomicMin(&val, 8) == 12);
    CHECK(val == 8);
    CHECK(cpu::atomicMax(&val, 20) == 8);
    CHECK(val == 20);
    CHECK(cpu::atomicExch(&val, 42) == 20);
    CHECK(val == 42);
    CHECK(cpu::atomicCAS(&val, 42, 99) == 42);
    CHECK(val == 99);
    CHECK(cpu::atomicCAS(&val, 42, 100) == 99);
    CHECK(val == 99);
    // and / or / xor used to be silently translated into atomicAdd by the CPU emitter
    val = 0xF0;
    CHECK(cpu::atomicAnd(&val, 0x3C) == 0xF0 && val == 0x30);
    CHECK(cpu::atomicOr(&val, 0x0F) == 0x30 && val == 0x3F);
    CHECK(cpu::atomicXor(&val, 0xFF) == 0x3F && val == 0xC0);
    unsigned int uval = 5u;
    CHECK(cpu::atomicMax(&uval, 9u) == 5u && uval == 9u);
    CHECK(cpu::atomicMin(&uval, 2u) == 9u && uval == 2u);

    float fval = 1.5f;
    CHECK(cpu::atomicAdd(&fval, 2.5f) == 1.5f);
    CHECK(fval == 4.0f);
    CHECK(cpu::atomicExch(&fval, 10.0f) == 4.0f);
    CHECK(fval == 10.0f);

    // 5. Buffer with custom struct
    TestParticle* p = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&p), 10 * sizeof(TestParticle)) == cudaSuccess);
    Launch l(table());
    CHECK(l.buffer<TestParticle>(0, p, Access::ReadWrite));
    CHECK(cudaFree(p) == cudaSuccess);
}

static void testSymbolBounds() {
    float symbol[4] = {1.f, 2.f, 3.f, 4.f};
    float src[8] = {};
    CHECK(cudaMemcpyToSymbol(symbol, src, sizeof(symbol)) == cudaSuccess);
    CHECK(cudaMemcpyToSymbol(symbol, src, sizeof(symbol) + 1) == cudaErrorInvalidValue);
    CHECK(cudagfx::lastErrorCode() == ErrorCode::OutOfBounds);
    CHECK(cudaMemcpyToSymbol(symbol, src, sizeof(float), sizeof(symbol) - sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpyToSymbol(symbol, src, sizeof(float), sizeof(symbol)) == cudaErrorInvalidValue);
    CHECK(cudaMemcpyToSymbol(symbol, src, 1, static_cast<size_t>(-1)) == cudaErrorInvalidValue); // offset + count overflows
    CHECK(cudaMemcpyFromSymbol(src, symbol, sizeof(symbol) + 4) == cudaErrorInvalidValue);
    CHECK(cudagfx::lastErrorCode() == ErrorCode::OutOfBounds);
    symbol[3] = 4.f; // (the first copy above overwrote the symbol with zeros)
    CHECK(cudaMemcpyFromSymbol(src, symbol, sizeof(symbol)) == cudaSuccess && src[3] == 4.f);
    cudaGetLastError();
}

// The generated host code registers every __constant__ object the shaders read; the runtime re-reads the object at each
// launch, so what cudaMemcpyToSymbol writes is what the next launch uploads.
static void testConstantRegistry() {
    static float object[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const cudagfx::ConstantRegistration registration("test_constant_object", &object, sizeof(object));
    const auto& constants = cudagfx::detail::state().constants;
    const auto it = constants.find("test_constant_object");
    CHECK(it != constants.end());
    if (it == constants.end()) return;
    CHECK(it->second.address == static_cast<const void*>(object) && it->second.bytes == sizeof(object));
    const float update[2] = {30.0f, 40.0f};
    CHECK(cudaMemcpyToSymbol(object, update, sizeof(update), 2 * sizeof(float)) == cudaSuccess);
    CHECK(static_cast<const float*>(it->second.address)[2] == 30.0f && static_cast<const float*>(it->second.address)[3] == 40.0f);
    CHECK(constants.find("not_registered") == constants.end());

    cudagfx::ConstantRef ref; // a default reference is empty
    CHECK(ref.slots == 0 && ref.bytes == 0 && ref.symbol[0] == '\0');
    cudagfx::ShaderBinaryTable table;
    CHECK(table.constants == nullptr && table.constantCount == 0);
}

static void testErrorCodeNames() {
    CHECK(std::strcmp(cudagfx::errorCodeName(ErrorCode::MissingConstant), "CGFXR012") == 0);
    CHECK(std::strcmp(cudaGetErrorString(cudaErrorInvalidSymbol), "cudaErrorInvalidSymbol") == 0);
    CHECK(std::strcmp(cudagfx::errorCodeName(ErrorCode::NoBackend), "CGFXR009") == 0);
    CHECK(std::strcmp(cudagfx::errorCodeName(ErrorCode::BackendFailure), "CGFXR010") == 0);
    CHECK(std::strcmp(cudagfx::errorCodeName(ErrorCode::ElementLayout), "CGFXR011") == 0);
    CHECK(std::strcmp(cudaGetErrorString(cudaErrorNotSupported), "cudaErrorNotSupported") == 0);
}

#if !CUDAGFX_HAS_BGFX
// Without bgfx a launch must report an error rather than pretend that the kernel ran.
static void testLaunchWithoutBackend() {
    cudaGetLastError();
    float* d = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&d), 256 * sizeof(float)) == cudaSuccess);
    Launch l(table());
    CHECK(l.buffer<float>(0, d, Access::ReadWrite));
    l.dispatch(dim3(1), dim3(64));
    CHECK(cudaGetLastError() != cudaSuccess);
    CHECK(cudaFree(d) == cudaSuccess);
}
#endif

// ---- CPU backend: executor, barrier, vectors -------------------------------------------------------

namespace cpu = cudagfx::cpu;

static void testBlockBarrier() {
    // A thread that leaves the kernel must not hold up the others (`if (i >= n) return;` before a barrier).
    cpu::BlockBarrier b(3);
    int released = 0;
    std::thread t1([&] { b.wait(); __atomic_fetch_add(&released, 1, __ATOMIC_SEQ_CST); });
    std::thread t2([&] { b.wait(); __atomic_fetch_add(&released, 1, __ATOMIC_SEQ_CST); });
    b.drop(); // the third participant exits without ever reaching the barrier
    t1.join();
    t2.join();
    CHECK(released == 2);
}

struct NoShared {};

static void testRunKernelNoBarrier() {
    std::vector<int> out(3 * 4 * 5, -1);
    cpu::runKernel<NoShared>(dim3(3), dim3(4, 5), false, [&](NoShared&) {
        const unsigned block = cpu::blockIdx.x;
        const unsigned thread = cpu::threadIdx.y * cpu::blockDim.x + cpu::threadIdx.x;
        out[block * 20 + thread] = static_cast<int>(block * 100 + thread);
    });
    bool ok = true;
    for (unsigned b = 0; b < 3; ++b) {
        for (unsigned t = 0; t < 20; ++t) ok = ok && out[b * 20 + t] == static_cast<int>(b * 100 + t);
    }
    CHECK(ok);
    // a zero-sized launch does nothing
    cpu::runKernel<NoShared>(dim3(1), dim3(0), false, [&](NoShared&) { CHECK(false); });
}

struct ReduceShared {
    int data[64];
};

// Block reduction with __syncthreads() in a loop, several blocks (the shared memory must be fresh for each block),
// and threads past the end of the data that return before the barriers.
static void testRunKernelWithBarrier() {
    const unsigned n = 64 * 3 - 10; // the last block is not full
    std::vector<int> in(n, 1), out(3, 0);
    cpu::runKernel<ReduceShared>(dim3(3), dim3(64), true, [&](ReduceShared& sh) {
        const unsigned tid = cpu::threadIdx.x;
        const unsigned i = cpu::blockIdx.x * cpu::blockDim.x + tid;
        if (i >= n) return; // leaves before the barriers below
        sh.data[tid] = in[i];
        cpu::syncthreads();
        for (unsigned s = 32; s > 0; s /= 2) {
            if (tid < s && i + s < n) sh.data[tid] += sh.data[tid + s];
            cpu::syncthreads();
        }
        if (tid == 0) out[cpu::blockIdx.x] = sh.data[0];
    });
    CHECK(out[0] == 64);
    CHECK(out[1] == 64);
    CHECK(out[2] == 64 - 10);
}

static void testVectorsAndSwizzles() {
    float4 a = make_float4(1.f, 2.f, 3.f, 4.f);
    float4 b = make_float4(10.f, 20.f, 30.f, 40.f);
    float4 sum = a + b;
    CHECK(sum.x == 11.f && sum.w == 44.f);
    float4 scaled = a * 2.f;
    CHECK(scaled.z == 6.f);
    float4 scaled2 = 2.f * a;
    CHECK(scaled2.y == 4.f);
    float4 diff = b - a;
    CHECK(diff.y == 18.f);
    float4 quot = b / 10.f;
    CHECK(quot.w == 4.f);
    float4 neg = -a;
    CHECK(neg.x == -1.f && neg.w == -4.f);
    int2 ia = make_int2(3, 4);
    int2 ib = ia * 2;
    CHECK(ib.x == 6 && ib.y == 8);
    a += b;
    CHECK(a.x == 11.f);

    float4 c = make_float4(1.f, 2.f, 3.f, 4.f);
    const float2 zw = cpu::swz<'z', 'w'>(c);
    CHECK(zw.x == 3.f && zw.y == 4.f);
    const float3 zyx = cpu::swz<'z', 'y', 'x'>(c);
    CHECK(zyx.x == 3.f && zyx.z == 1.f);
    cpu::swizzleRef<'x', 'y'>(c) = cpu::swz<'z', 'w'>(c) * 0.5f; // c.xy = c.zw * 0.5f
    CHECK(c.x == 1.5f && c.y == 2.f && c.z == 3.f);
    cpu::swizzleRef<'z', 'w'>(c) += make_float2(1.f, 1.f);
    CHECK(c.z == 4.f && c.w == 5.f);
    CHECK(cpu::comp(c, 3) == 5.f);
    cpu::comp(c, 0) = 9.f;
    CHECK(c.x == 9.f);
}

int main() {
    testAllocation();
    testCopies();
    testErrorState();
    testLaunchValidation();
    testPhase6Features();
    testSymbolBounds();
    testConstantRegistry();
    testErrorCodeNames();
#if !CUDAGFX_HAS_BGFX
    testLaunchWithoutBackend();
#endif
    testBlockBarrier();
    testRunKernelNoBarrier();
    testRunKernelWithBarrier();
    testVectorsAndSwizzles();
    cudagfx::shutdown();
    std::printf(g_failures ? "FAILURES: %d/%d\n" : "OK: %d checks, %d failure(s)\n", g_failures ? g_failures : g_checks, g_failures ? g_checks : g_failures);
    return g_failures == 0 ? 0 : 1;
}

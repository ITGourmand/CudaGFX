// Test of the runtime WITHOUT a GPU (CUDAGFX_NO_BGFX): memory, copies, bounds, launch validation, error codes.
// Compiled with -fno-exceptions -fno-rtti to check the runtime constraint C12.
#include <cstdio>
#include <cstring>
#include <vector>

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

int main() {
    testAllocation();
    testCopies();
    testErrorState();
    testLaunchValidation();
    cudagfx::shutdown();
    std::printf(g_failures ? "FAILURES: %d/%d\n" : "OK: %d checks, %d failure(s)\n", g_failures ? g_failures : g_checks, g_failures ? g_checks : g_failures);
    return g_failures == 0 ? 0 : 1;
}

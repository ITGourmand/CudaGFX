// main_handwritten.cpp: "Phase 0" validation of the runtime with a handwritten SAXPY shader.
//
// This program does NOT depend on the transpiler: it checks that the runtime, bgfx and shaderc agree with each
// other (uniforms, buffers, argument ABI). The cs_saxpy_*.h binaries were produced by shaderc from
// cs_saxpy.sc. Without bgfx (CUDAGFX_NO_BGFX) only the memory part can be verified.
//
// Usage: call bgfx::init(...) first and then cudagfx::init(), as in a real application; this file provides a
// minimal main() that assumes bgfx has already been initialized by the host application.
#include <cmath>
#include <cstdio>
#include <vector>

#include "CudaGFX/cudagfx_rt.h"
#include "cs_saxpy_430.h"
#include "cs_saxpy_metal.h"
#include "cs_saxpy_spv.h"
#include "cs_saxpy_wgsl.h"

namespace {

// Replaces assert(): it does not disappear with -DNDEBUG and it prints the cudagfx cause of the failure.
int g_failures = 0;
void check(bool ok, const char* what) {
    if (ok) return;
    ++g_failures;
    std::fprintf(stderr, "FAILED: %s", what);
    if (cudagfx::lastErrorCode() != cudagfx::ErrorCode::None) {
        std::fprintf(stderr, " [%s] %s", cudagfx::errorCodeName(cudagfx::lastErrorCode()), cudagfx::lastErrorMessage());
    }
    std::fprintf(stderr, "\n");
}

// Describes the compiled shader exactly like the generated *_kernels.h headers do.
cudagfx::ShaderBinaryTable saxpyTable() {
    cudagfx::ShaderBinaryTable table;
    table.kernel = "saxpy";
    table.argsUniform = "u_saxpy_args"; // name declared by cs_saxpy.sc
    table.gridUniform = "";
    table.block[0] = 256;
    table.spv = {cs_saxpy_spv, sizeof(cs_saxpy_spv)};
    table.glsl = {cs_saxpy_430, sizeof(cs_saxpy_430)};
    table.mtl = {cs_saxpy_metal, sizeof(cs_saxpy_metal)};
    table.wgsl = {cs_saxpy_wgsl, sizeof(cs_saxpy_wgsl)};
    return table;
}

// Equivalent of what the generated stub does: pack the by-value arguments, bind the buffers, dispatch.
void saxpy(dim3 grid, dim3 blk, float a, const float* x, float* y, int n) {
    cudagfx::Launch l(saxpyTable());
    l.arg(a);
    l.arg(n);
    l.template buffer<float>(0, x, cudagfx::Access::Read);
    l.template buffer<float>(1, y, cudagfx::Access::ReadWrite);
    l.dispatch(grid, blk);
}

} // namespace

int main() {
    std::printf("[Phase 0] handwritten SAXPY\n");
    cudagfx::init();

    const int n = 1000;
    const float a = 2.5f;
    std::vector<float> hx(n), hy(n), expected(n);
    for (int i = 0; i < n; ++i) {
        hx[i] = static_cast<float>(i);
        hy[i] = 10.0f;
        expected[i] = a * hx[i] + hy[i];
    }

    float *dx = nullptr, *dy = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&dx), n * sizeof(float)) == cudaSuccess, "cudaMalloc(dx)");
    check(cudaMalloc(reinterpret_cast<void**>(&dy), n * sizeof(float)) == cudaSuccess, "cudaMalloc(dy)");
    check(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess, "cudaMemcpy x");
    check(cudaMemcpy(dy, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess, "cudaMemcpy y");

    saxpy(dim3((n + 255) / 256), dim3(256), a, dx, dy, n);
    check(cudaGetLastError() == cudaSuccess, "kernel launch");

    check(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess, "cudaMemcpy result");

    int wrong = 0;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(hy[i] - expected[i]) > 1e-4f) ++wrong;
    }
    check(wrong == 0, "incorrect SAXPY result");

    check(cudaFree(dx) == cudaSuccess, "cudaFree(dx)");
    check(cudaFree(dy) == cudaSuccess, "cudaFree(dy)");
    cudagfx::shutdown();

    std::printf(g_failures == 0 ? "[Phase 0] OK: %d elements verified\n" : "[Phase 0] %d failure(s)\n", g_failures == 0 ? n : g_failures);
    return g_failures == 0 ? 0 : 1;
}

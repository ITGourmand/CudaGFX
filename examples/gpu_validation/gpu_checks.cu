// GPU validation: kernels that exercise what has never been run on a device, each one compared with a host reference.
//
//   gv_saxpy   by-value arguments (uniform), float buffers
//   integrate  a 32-byte struct buffer: the stride of the bgfx buffer must match the shader's
//   block_sum  shared memory, __syncthreads(), a loop with barriers, gridDim
//   histogram  integer atomics (add / or / max)
//   swizzle    vector arithmetic and swizzles, float4 buffers
//   constants  __constant__ memory: uniform arrays refilled from the host objects, updated between launches
//
// The same checks run on the CPU backend, so a failure that only shows on the GPU points at the GPU path.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gpu_checks.h"
#include "gv_common.h"

#define BLOCK 256

struct Particle {
    float4 pos;
    float4 vel;
};

__global__ void gv_saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

__global__ void integrate(Particle* p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i].pos.x += p[i].vel.x * dt;
    p[i].pos.y += p[i].vel.y * dt;
    p[i].pos.z += p[i].vel.z * dt;
    p[i].pos.w = (float)i;
}

__global__ void block_sum(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = sdata[tid] + sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void histogram(const int* values, int* bins, int* maxValue, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&bins[values[i] & 15], 1);
    atomicOr(&bins[16], 1 << (values[i] & 15));
    atomicMax(&maxValue[0], values[i]);
}

__global__ void swizzle(float4* y, const float4* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 v = x[i];
    float4 r = v * 2.0f + v;
    r.xy = r.zw * 0.5f;
    r.w = v.x - v.y;
    y[i] = r;
}

__constant__ float c_gain = 1.5f;
__constant__ int c_bias = 7;
__constant__ float4 c_colors[4];
__constant__ float3 c_dirs[2]; // 12-byte elements: they straddle the vec4 slots of the uniform array
__constant__ uint2 c_ids[3];
__constant__ float c_table[300]; // 75 vec4 slots

__global__ void gv_constants(float* y, int* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int u = (unsigned int)i;
    y[i] = c_colors[u & 3u].y * c_gain + c_dirs[u & 1u].z + c_table[u % 300u];
    z[i] = c_bias + (int)c_ids[u % 3u].y;
}

namespace {

using gv::close;
using gv::download;
using gv::upload;

bool checkSaxpy() {
    const int n = 1000003; // not a multiple of the block size
    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i % 101);
        y[i] = static_cast<float>(i % 13);
    }
    float* dx = upload(x);
    float* dy = upload(y);
    gv_saxpy<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(2.5f, dx, dy, n);
    std::vector<float> out(n);
    download(out, dy);
    cudaFree(dx);
    cudaFree(dy);
    for (int i = 0; i < n; ++i) {
        if (!close(out[i], 2.5f * x[i] + y[i])) {
            std::printf("    saxpy[%d] = %f, expected %f\n", i, out[i], 2.5f * x[i] + y[i]);
            return false;
        }
    }
    return true;
}

bool checkIntegrate() {
    const int n = 5000;
    std::vector<Particle> p(n);
    for (int i = 0; i < n; ++i) {
        p[i].pos = make_float4(1.0f * i, 2.0f * i, 3.0f * i, -1.0f);
        p[i].vel = make_float4(0.5f, -0.25f, 2.0f, 9.0f);
    }
    Particle* d = upload(p);
    integrate<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d, 0.1f, n);
    std::vector<Particle> out(n);
    download(out, d);
    cudaFree(d);
    for (int i = 0; i < n; ++i) {
        const bool ok = close(out[i].pos.x, 1.0f * i + 0.05f) && close(out[i].pos.y, 2.0f * i - 0.025f) &&
                        close(out[i].pos.z, 3.0f * i + 0.2f) && out[i].pos.w == static_cast<float>(i) &&
                        close(out[i].vel.x, 0.5f) && close(out[i].vel.w, 9.0f); // the velocities must be left intact
        if (!ok) {
            std::printf("    particle %d: pos (%f %f %f %f) vel (%f %f %f %f)\n", i, out[i].pos.x, out[i].pos.y, out[i].pos.z, out[i].pos.w,
                        out[i].vel.x, out[i].vel.y, out[i].vel.z, out[i].vel.w);
            return false;
        }
    }
    return true;
}

bool checkBlockSum() {
    const int n = BLOCK * 37 + 11;
    const int blocks = (n + BLOCK - 1) / BLOCK;
    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) in[i] = static_cast<float>((i * 7) % 5);
    std::vector<float> out(blocks, -1.0f);
    float* din = upload(in);
    float* dout = upload(out);
    block_sum<<<blocks, BLOCK>>>(din, dout, n);
    download(out, dout);
    cudaFree(din);
    cudaFree(dout);
    for (int b = 0; b < blocks; ++b) {
        float expected = 0.0f;
        for (int i = b * BLOCK; i < n && i < (b + 1) * BLOCK; ++i) expected += in[i];
        if (!close(out[b], expected)) {
            std::printf("    block %d: %f, expected %f\n", b, out[b], expected);
            return false;
        }
    }
    return true;
}

bool checkHistogram() {
    const int n = 10000;
    std::vector<int> values(n);
    for (int i = 0; i < n; ++i) values[i] = (i * 37 + 5) % 1000;
    std::vector<int> bins(17, 0), maxValue(1, 0);
    int* dv = upload(values);
    int* db = upload(bins);
    int* dm = upload(maxValue);
    histogram<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(dv, db, dm, n);
    download(bins, db);
    download(maxValue, dm);
    cudaFree(dv);
    cudaFree(db);
    cudaFree(dm);
    std::vector<int> expected(17, 0);
    int expectedMax = 0;
    for (int v : values) {
        expected[v & 15] += 1;
        expected[16] |= 1 << (v & 15);
        if (v > expectedMax) expectedMax = v;
    }
    for (int b = 0; b < 17; ++b) {
        if (bins[b] != expected[b]) {
            std::printf("    bin %d: %d, expected %d\n", b, bins[b], expected[b]);
            return false;
        }
    }
    if (maxValue[0] != expectedMax) {
        std::printf("    max: %d, expected %d\n", maxValue[0], expectedMax);
        return false;
    }
    return true;
}

bool checkSwizzle() {
    const int n = 3000;
    std::vector<float4> x(n), y(n);
    for (int i = 0; i < n; ++i) x[i] = make_float4(1.0f + i, 2.0f * i, 3.0f - i, 0.5f * i);
    float4* dx = upload(x);
    float4* dy = upload(y);
    swizzle<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(dy, dx, n);
    download(y, dy);
    cudaFree(dx);
    cudaFree(dy);
    for (int i = 0; i < n; ++i) {
        const float ex = 3.0f * x[i].z * 0.5f, ey = 3.0f * x[i].w * 0.5f;
        const float ez = 3.0f * x[i].z, ew = x[i].x - x[i].y;
        if (!(close(y[i].x, ex) && close(y[i].y, ey) && close(y[i].z, ez) && close(y[i].w, ew))) {
            std::printf("    swizzle[%d] = (%f %f %f %f)\n", i, y[i].x, y[i].y, y[i].z, y[i].w);
            return false;
        }
    }
    return true;
}

bool launchConstants(const char* round) {
    const int n = 300;
    float* dy = gv::upload(std::vector<float>(n, 0.0f));
    int* dz = gv::upload(std::vector<int>(n, 0));
    gv_constants<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(dy, dz, n);
    std::vector<float> y(n);
    std::vector<int> z(n);
    gv::download(y, dy);
    gv::download(z, dz);
    cudaFree(dy);
    cudaFree(dz);
    for (int i = 0; i < n; ++i) {
        const unsigned int u = static_cast<unsigned int>(i);
        const float ey = c_colors[u & 3u].y * c_gain + c_dirs[u & 1u].z + c_table[u % 300u]; // the host objects ARE the truth
        const int ez = c_bias + static_cast<int>(c_ids[u % 3u].y);
        if (!close(y[i], ey) || z[i] != ez) {
            std::printf("    constants (%s) [%d]: (%f, %d), expected (%f, %d)\n", round, i, y[i], z[i], ey, ez);
            return false;
        }
    }
    return true;
}

bool checkConstants() {
    if (!launchConstants("initial values")) return false;

    float colors[16];
    float dirs[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    unsigned int ids[6] = {10u, 11u, 12u, 13u, 14u, 15u};
    float table[300];
    for (int i = 0; i < 16; ++i) colors[i] = 0.25f * static_cast<float>(i);
    for (int i = 0; i < 300; ++i) table[i] = static_cast<float>(i % 17) - 8.0f;
    const float gain = -0.5f;
    const int bias = 123456;
    cudaMemcpyToSymbol(c_colors, colors, sizeof(colors));
    cudaMemcpyToSymbol(c_dirs, dirs, sizeof(dirs));
    cudaMemcpyToSymbol(c_ids, ids, sizeof(ids));
    cudaMemcpyToSymbol(c_table, table, sizeof(table));
    cudaMemcpyToSymbol(c_gain, &gain, sizeof(gain));
    cudaMemcpyToSymbol(c_bias, &bias, sizeof(bias));
    if (!launchConstants("after cudaMemcpyToSymbol")) return false;

    // A partial update (offset) must reach the shader too.
    const float one = 100.0f;
    cudaMemcpyToSymbol(c_table, &one, sizeof(one), 150 * sizeof(float));
    const int negative = -42;
    cudaMemcpyToSymbol(c_bias, &negative, sizeof(negative));
    return launchConstants("after a partial update");
}

} // namespace

int runGpuChecks(bool useGpu) {
    const gv::Check checks[] = {{"saxpy", checkSaxpy},
                                {"integrate (32-byte struct buffer)", checkIntegrate},
                                {"block_sum (shared memory + barriers)", checkBlockSum},
                                {"histogram (atomics)", checkHistogram},
                                {"swizzle (vectors)", checkSwizzle},
                                {"constants (uniform arrays, updated)", checkConstants}};
    return gv::runChecks("basics", checks, 6, useGpu);
}

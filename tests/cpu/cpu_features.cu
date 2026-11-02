// CPU backend: switch, vector arithmetic and swizzles (read and write), atomics (and/or/xor were once silently
// translated into add), a struct buffer, constants, __constant__ memory and device functions.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

__constant__ float c_w[4] = {1.0f, 2.0f, 3.0f, 4.0f};

struct Particle {
    float4 pos;
    float4 vel;
};

__device__ float lerp1(float a, float b, float t) { return a + (b - a) * t; }

__global__ void vec_kernel(float4* y, const float4* x, int n) {
    int i = threadIdx.x;
    if (i >= n) return;
    float4 v = x[i];
    float4 r = v * 2.0f + v;     // 3 v
    r.xy = r.zw * 0.5f;          // swizzle on both sides
    r.zw.x = r.z + 1.0f;         // swizzle of a swizzle, as an lvalue
    float2 t = r.yx;
    r.w = t.x - t.y;
    y[i] = r;
}

__global__ void switch_kernel(int* out, int n) {
    int i = threadIdx.x;
    if (i >= n) return;
    int r = 0;
    switch (i % 4) {
        case 0: r = 10; break;
        case 1: r = 20;
        case 2: r = r + 5; break;
        default: r = -1; break;
    }
    out[i] = r;
}

__global__ void atomics_kernel(int* bits, unsigned int* counter, int n) {
    int i = threadIdx.x;
    if (i >= n) return;
    atomicOr(&bits[0], 1 << (i & 31));
    atomicXor(&bits[1], 1 << (i & 7));
    atomicAnd(&bits[2], ~(1 << (i & 31)));
    atomicAdd(&counter[0], 1u);
}

__global__ void particles_kernel(Particle* p, float dt, int n) {
    int i = threadIdx.x;
    if (i >= n) return;
    p[i].pos.x += p[i].vel.x * dt;
    p[i].pos.y += p[i].vel.y * dt;
    p[i].pos.w = c_w[(unsigned)i % 4u] * lerp1(0.0f, 10.0f, 0.5f);
}

static int g_bad = 0;
static void check(bool ok, int line) {
    if (!ok) {
        std::printf("FAILED line %d\n", line);
        ++g_bad;
    }
}
#define CHECK(c) check((c), __LINE__)

int main() {
    cudagfx::setExecutionMode(cudagfx::ExecutionMode::CPU);
    {   // vectors and swizzles
        const int n = 8;
        std::vector<float4> x(n), y(n);
        for (int i = 0; i < n; ++i) x[i] = make_float4(1.f + i, 2.f + i, 3.f + i, 4.f + i);
        float4 *dx = nullptr, *dy = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&dx), n * sizeof(float4));
        cudaMalloc(reinterpret_cast<void**>(&dy), n * sizeof(float4));
        cudaMemcpy(dx, x.data(), n * sizeof(float4), cudaMemcpyHostToDevice);
        vec_kernel<<<1, 64>>>(dy, dx, n);
        cudaMemcpy(y.data(), dy, n * sizeof(float4), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            float vx = 3.f * x[i].x, vy = 3.f * x[i].y, vz = 3.f * x[i].z, vw = 3.f * x[i].w;
            (void)vx; (void)vy;
            const float ex = vz * 0.5f, ey = vw * 0.5f;     // xy = zw * 0.5
            const float ez = vz + 1.0f;                     // z = z + 1 (via .zw.x)
            const float ew = ey - ex;                       // w = t.x - t.y with t = (y, x)
            CHECK(y[i].x == ex && y[i].y == ey && y[i].z == ez && y[i].w == ew);
        }
        cudaFree(dx);
        cudaFree(dy);
    }

    {   // switch with fall-through
        const int n = 16;
        std::vector<int> out(n, 0);
        int* d = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&d), n * sizeof(int));
        switch_kernel<<<1, 64>>>(d, n);
        cudaMemcpy(out.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            const int e = (i % 4 == 0) ? 10 : (i % 4 == 1) ? 25 : (i % 4 == 2) ? 5 : -1;
            CHECK(out[i] == e);
        }
        cudaFree(d);
    }

    {   // atomics
        const int n = 64;
        int hostBits[3] = {0, 0, -1};
        unsigned int hostCounter = 0;
        int* dBits = nullptr;
        unsigned int* dCounter = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&dBits), sizeof(hostBits));
        cudaMalloc(reinterpret_cast<void**>(&dCounter), sizeof(hostCounter));
        cudaMemcpy(dBits, hostBits, sizeof(hostBits), cudaMemcpyHostToDevice);
        cudaMemcpy(dCounter, &hostCounter, sizeof(hostCounter), cudaMemcpyHostToDevice);
        atomics_kernel<<<1, 64>>>(dBits, dCounter, n);
        cudaMemcpy(hostBits, dBits, sizeof(hostBits), cudaMemcpyDeviceToHost);
        cudaMemcpy(&hostCounter, dCounter, sizeof(hostCounter), cudaMemcpyDeviceToHost);
        CHECK(hostBits[0] == -1);   // every bit set by an or
        CHECK(hostBits[1] == 0);    // each of the 8 low bits toggled 8 times
        CHECK(hostBits[2] == 0);    // every bit cleared by an and
        CHECK(hostCounter == 64u);
        cudaFree(dBits);
        cudaFree(dCounter);
    }

    {   // struct buffer, __constant__ memory, device function
        const int n = 6;
        std::vector<Particle> p(n);
        for (int i = 0; i < n; ++i) {
            p[i].pos = make_float4(1.f, 2.f, 3.f, 0.f);
            p[i].vel = make_float4(10.f * i, 20.f * i, 0.f, 0.f);
        }
        Particle* d = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&d), n * sizeof(Particle));
        cudaMemcpy(d, p.data(), n * sizeof(Particle), cudaMemcpyHostToDevice);
        particles_kernel<<<1, 64>>>(d, 0.5f, n);
        std::vector<Particle> q(n);
        cudaMemcpy(q.data(), d, n * sizeof(Particle), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            CHECK(q[i].pos.x == 1.f + 10.f * i * 0.5f);
            CHECK(q[i].pos.y == 2.f + 20.f * i * 0.5f);
            CHECK(q[i].pos.z == 3.f);
            CHECK(q[i].pos.w == c_w[i % 4] * 5.0f);
        }
        cudaFree(d);
    }

    if (cudaGetLastError() != cudaSuccess) {
        std::printf("runtime error: %s\n", cudagfx::lastErrorMessage());
        return 2;
    }
    std::printf(g_bad ? "FAILED\n" : "OK\n");
    return g_bad ? 1 : 0;
}

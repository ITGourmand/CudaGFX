// Classic GPU algorithms, each one compared with a host implementation:
//
//   Gray-Scott      reaction-diffusion stencil on float2 cells, 60 chained ping-pong generations
//   bitonic sort    shared memory, barriers inside two nested loops, unsigned integer logic (exact)
//   prefix scan     Hillis-Steele scan per block + scan of the block sums + offsets: three dependent kernels
//   tiled matmul    two shared tiles, 2D blocks, two barriers per tile
//   Mandelbrot      data-dependent loop with an early exit, one thread per pixel
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gpu_checks.h"
#include "gv_common.h"

#define SORT_N 256
#define SCAN_N 256
#define MM_TILE 16
#define MM_TILEU 16u

__global__ void gs_step(const float2* in, float2* out, int w, int h, float du, float dv, float feed, float kill, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    unsigned int uw = (unsigned int)w;
    unsigned int uh = (unsigned int)h;
    unsigned int ux = (unsigned int)x;
    unsigned int uy = (unsigned int)y;
    unsigned int xm = (ux + uw - 1u) % uw;
    unsigned int xp = (ux + 1u) % uw;
    unsigned int ym = (uy + uh - 1u) % uh;
    unsigned int yp = (uy + 1u) % uh;
    float2 c = in[uy * uw + ux];
    float2 lap = in[uy * uw + xm] + in[uy * uw + xp] + in[ym * uw + ux] + in[yp * uw + ux] - 4.0f * c;
    float uvv = c.x * c.y * c.y;
    float nu = c.x + dt * (du * lap.x - uvv + feed * (1.0f - c.x));
    float nv = c.y + dt * (dv * lap.y + uvv - (feed + kill) * c.y);
    out[uy * uw + ux] = make_float2(nu, nv);
}

__global__ void bitonic_sort(unsigned int* data) {
    __shared__ unsigned int s[SORT_N];
    unsigned int tid = threadIdx.x;
    unsigned int base = blockIdx.x * (unsigned int)SORT_N;
    s[tid] = data[base + tid];
    __syncthreads();
    for (unsigned int k = 2u; k <= (unsigned int)SORT_N; k <<= 1u) {
        for (unsigned int j = k >> 1u; j > 0u; j >>= 1u) {
            unsigned int partner = tid ^ j;
            if (partner > tid) {
                bool up = (tid & k) == 0u;
                unsigned int a = s[tid];
                unsigned int b = s[partner];
                if ((a > b) == up) {
                    s[tid] = b;
                    s[partner] = a;
                }
            }
            __syncthreads();
        }
    }
    data[base + tid] = s[tid];
}

__global__ void scan_blocks(const unsigned int* in, unsigned int* out, unsigned int* sums, int n) {
    __shared__ unsigned int tmp[SCAN_N];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    tmp[tid] = (gid < (unsigned int)n) ? in[gid] : 0u;
    __syncthreads();
    for (unsigned int offset = 1u; offset < (unsigned int)SCAN_N; offset <<= 1u) {
        unsigned int v = tmp[tid];
        if (tid >= offset) v = v + tmp[tid - offset];
        __syncthreads();
        tmp[tid] = v;
        __syncthreads();
    }
    if (gid < (unsigned int)n) out[gid] = tmp[tid];
    if (tid == (unsigned int)(SCAN_N - 1)) sums[blockIdx.x] = tmp[tid];
}

__global__ void scan_add(unsigned int* data, const unsigned int* offsets, int n) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (unsigned int)n) return;
    if (blockIdx.x > 0u) data[gid] = data[gid] + offsets[blockIdx.x - 1u];
}

__global__ void mm_tiled(const float* a, const float* b, float* c, int n) {
    __shared__ float as[MM_TILE * MM_TILE];
    __shared__ float bs[MM_TILE * MM_TILE];
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int row = blockIdx.y * MM_TILEU + ty;
    unsigned int col = blockIdx.x * MM_TILEU + tx;
    unsigned int un = (unsigned int)n;
    float acc = 0.0f;
    for (unsigned int t = 0u; t < un / MM_TILEU; ++t) {
        as[ty * MM_TILEU + tx] = a[row * un + t * MM_TILEU + tx];
        bs[ty * MM_TILEU + tx] = b[(t * MM_TILEU + ty) * un + col];
        __syncthreads();
        for (unsigned int k = 0u; k < MM_TILEU; ++k) acc = acc + as[ty * MM_TILEU + k] * bs[k * MM_TILEU + tx];
        __syncthreads();
    }
    c[row * un + col] = acc;
}

__global__ void mandel(unsigned int* out, int w, int h, int maxIter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float cx = -2.2f + 3.2f * ((float)x / (float)w);
    float cy = -1.2f + 2.4f * ((float)y / (float)h);
    float zx = 0.0f;
    float zy = 0.0f;
    int it = 0;
    while (it < maxIter && zx * zx + zy * zy <= 4.0f) {
        float t = zx * zx - zy * zy + cx;
        zy = 2.0f * zx * zy + cy;
        zx = t;
        it = it + 1;
    }
    out[y * w + x] = (unsigned int)it;
}

namespace {

bool checkGrayScott() {
    const int w = 70, h = 50, steps = 60; // not a multiple of the block
    const float du = 0.16f, dv = 0.08f, feed = 0.0367f, kill = 0.0649f, dt = 1.0f;
    std::vector<float2> init(static_cast<size_t>(w) * h);
    uint32_t seed = 5u;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const bool center = x >= w / 2 - 5 && x < w / 2 + 5 && y >= h / 2 - 5 && y < h / 2 + 5;
            const float noise = 0.01f * gv::random01(seed);
            init[y * w + x] = center ? make_float2(0.5f + noise, 0.25f + noise) : make_float2(1.0f, noise);
        }
    }

    std::vector<float2> a = init, b = init;
    for (int s = 0; s < steps; ++s) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int xm = (x + w - 1) % w, xp = (x + 1) % w, ym = (y + h - 1) % h, yp = (y + 1) % h;
                const float2 c = a[y * w + x];
                const float lapx = a[y * w + xm].x + a[y * w + xp].x + a[ym * w + x].x + a[yp * w + x].x - 4.0f * c.x;
                const float lapy = a[y * w + xm].y + a[y * w + xp].y + a[ym * w + x].y + a[yp * w + x].y - 4.0f * c.y;
                const float uvv = c.x * c.y * c.y;
                b[y * w + x] = make_float2(c.x + dt * (du * lapx - uvv + feed * (1.0f - c.x)), c.y + dt * (dv * lapy + uvv - (feed + kill) * c.y));
            }
        }
        a.swap(b);
    }

    float2* dIn = gv::upload(init);
    float2* dOut = gv::upload(init);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    for (int s = 0; s < steps; ++s) {
        gs_step<<<grid, dim3(16, 16)>>>(dIn, dOut, w, h, du, dv, feed, kill, dt);
        float2* t = dIn;
        dIn = dOut;
        dOut = t;
    }
    std::vector<float2> got(init.size());
    gv::download(got, dIn);
    cudaFree(dIn);
    cudaFree(dOut);
    for (size_t i = 0; i < got.size(); ++i) {
        if (!gv::close(got[i].x, a[i].x, 1e-3f) || !gv::close(got[i].y, a[i].y, 1e-3f)) {
            std::printf("    cell %d: (%f, %f), expected (%f, %f)\n", static_cast<int>(i), got[i].x, got[i].y, a[i].x, a[i].y);
            return false;
        }
    }
    return true;
}

bool checkBitonicSort() {
    const int blocks = 16, n = blocks * SORT_N;
    std::vector<unsigned int> data(n);
    uint32_t seed = 77u;
    for (auto& v : data) v = gv::nextRandom(seed) >> 8;
    unsigned int* d = gv::upload(data);
    bitonic_sort<<<blocks, SORT_N>>>(d);
    std::vector<unsigned int> got(n);
    gv::download(got, d);
    cudaFree(d);
    for (int b = 0; b < blocks; ++b) {
        std::vector<unsigned int> expected(data.begin() + b * SORT_N, data.begin() + (b + 1) * SORT_N);
        std::sort(expected.begin(), expected.end());
        for (int i = 0; i < SORT_N; ++i) {
            if (got[b * SORT_N + i] != expected[i]) {
                std::printf("    block %d, element %d: %u, expected %u\n", b, i, got[b * SORT_N + i], expected[i]);
                return false;
            }
        }
    }
    return true;
}

bool checkScan() {
    const int n = 4000; // 16 blocks, the last one partial
    const int blocks = (n + SCAN_N - 1) / SCAN_N;
    std::vector<unsigned int> data(n);
    uint32_t seed = 9u;
    for (auto& v : data) v = gv::nextRandom(seed) >> 24; // 0..255
    unsigned int* dIn = gv::upload(data);
    unsigned int* dOut = gv::upload(std::vector<unsigned int>(n, 0u));
    unsigned int* dSums = gv::upload(std::vector<unsigned int>(SCAN_N, 0u));
    unsigned int* dSumsScanned = gv::upload(std::vector<unsigned int>(SCAN_N, 0u));
    unsigned int* dScratch = gv::upload(std::vector<unsigned int>(1, 0u));
    scan_blocks<<<blocks, SCAN_N>>>(dIn, dOut, dSums, n);                  // 1. scan of every block, block totals
    scan_blocks<<<1, SCAN_N>>>(dSums, dSumsScanned, dScratch, blocks);     // 2. scan of the block totals
    scan_add<<<blocks, SCAN_N>>>(dOut, dSumsScanned, n);                   // 3. add the offsets of the previous blocks
    std::vector<unsigned int> got(n);
    gv::download(got, dOut);
    cudaFree(dIn);
    cudaFree(dOut);
    cudaFree(dSums);
    cudaFree(dSumsScanned);
    cudaFree(dScratch);
    unsigned int running = 0;
    for (int i = 0; i < n; ++i) {
        running += data[i];
        if (got[i] != running) {
            std::printf("    prefix %d: %u, expected %u\n", i, got[i], running);
            return false;
        }
    }
    return true;
}

bool checkMatmul() {
    const int n = 64;
    std::vector<float> a(n * n), b(n * n), c(n * n, 0.0f);
    uint32_t seed = 3u;
    for (auto& v : a) v = gv::random01(seed) - 0.5f;
    for (auto& v : b) v = gv::random01(seed) - 0.5f;
    float* dA = gv::upload(a);
    float* dB = gv::upload(b);
    float* dC = gv::upload(c);
    mm_tiled<<<dim3(n / MM_TILE, n / MM_TILE), dim3(MM_TILE, MM_TILE)>>>(dA, dB, dC, n);
    gv::download(c, dC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k) acc = acc + a[row * n + k] * b[k * n + col];
            if (!gv::close(c[row * n + col], acc, 1e-4f)) {
                std::printf("    c[%d][%d] = %f, expected %f\n", row, col, c[row * n + col], acc);
                return false;
            }
        }
    }
    return true;
}

bool checkMandelbrot() {
    const int w = 160, h = 96, maxIter = 200;
    std::vector<unsigned int> expected(static_cast<size_t>(w) * h), got(expected.size(), 0u);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float cx = -2.2f + 3.2f * (static_cast<float>(x) / static_cast<float>(w));
            const float cy = -1.2f + 2.4f * (static_cast<float>(y) / static_cast<float>(h));
            float zx = 0.0f, zy = 0.0f;
            int it = 0;
            while (it < maxIter && zx * zx + zy * zy <= 4.0f) {
                const float t = zx * zx - zy * zy + cx;
                zy = 2.0f * zx * zy + cy;
                zx = t;
                it = it + 1;
            }
            expected[y * w + x] = static_cast<unsigned int>(it);
        }
    }
    unsigned int* d = gv::upload(got);
    mandel<<<dim3((w + 15) / 16, (h + 15) / 16), dim3(16, 16)>>>(d, w, h, maxIter);
    gv::download(got, d);
    cudaFree(d);
    // Pixels on the border of the set are sensitive to a one-ulp difference (fused multiply-add on a GPU): demand
    // that 99 % of the pixels are exactly the same.
    size_t same = 0;
    for (size_t i = 0; i < got.size(); ++i) same += got[i] == expected[i] ? 1u : 0u;
    if (same * 100 < got.size() * 99) {
        std::printf("    only %zu of %zu pixels match\n", same, got.size());
        return false;
    }
    return true;
}

} // namespace

int runAlgorithmChecks(bool useGpu) {
    const gv::Check checks[] = {{"gray-scott: 70x50, 60 generations (float2 stencil)", checkGrayScott},
                                {"bitonic sort: 16 blocks of 256", checkBitonicSort},
                                {"prefix scan: 3 dependent kernels, 4000 values", checkScan},
                                {"matmul: 64x64, shared tiles", checkMatmul},
                                {"mandelbrot: 160x96, early-exit loop", checkMandelbrot}};
    return gv::runChecks("algorithms", checks, 5, useGpu);
}

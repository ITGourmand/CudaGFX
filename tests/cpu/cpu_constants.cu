// __constant__ memory: the kernels read the CURRENT content of the host object, so cudaMemcpyToSymbol between two
// launches changes what the next launch computes (on the GPU too, where the object is re-uploaded at each launch).
// Scalars, vectors (float3 has a 12-byte stride), arrays, partial updates through an offset, device functions.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__constant__ float c_scale = 2.0f;
__constant__ int c_offset = 3;
__constant__ unsigned int c_mask = 0xFFu;
__constant__ float2 c_dir = {1.0f, 0.0f};
__constant__ float3 c_axes[2];
__constant__ float4 c_palette[4];
__constant__ int2 c_cells[3];
__constant__ uint4 c_bits[2];
__constant__ float c_lut[256];

__device__ float shade(unsigned int i) { return c_palette[i & 3u].x * c_scale + c_dir.y; }

__global__ void use_constants(float* y, int* z, unsigned int* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = shade((unsigned int)i) + c_axes[(unsigned int)i & 1u].y + c_lut[(unsigned int)i & 255u] + (float)c_cells[(unsigned int)i % 3u].x;
    z[i] = c_offset + c_cells[(unsigned int)i % 3u].y;
    w[i] = (c_bits[(unsigned int)i & 1u].z + (unsigned int)i) & c_mask;
}

static int g_bad = 0;
static void check(bool ok, int line) {
    if (!ok) {
        std::printf("FAILED line %d\n", line);
        ++g_bad;
    }
}
#define CHECK(c) check((c), __LINE__)

static const int N = 40;

// What the kernel computes from the current values of the constants, written independently on the host.
static void expected(std::vector<float>& y, std::vector<int>& z, std::vector<unsigned int>& w) {
    float palette[4][4], axes[2][3], lut[256];
    int cells[3][2];
    unsigned int bits[2][4];
    float scale, dir[2];
    int offset;
    unsigned int mask;
    cudaMemcpyFromSymbol(palette, c_palette, sizeof(palette));
    cudaMemcpyFromSymbol(axes, c_axes, sizeof(axes));
    cudaMemcpyFromSymbol(lut, c_lut, sizeof(lut));
    cudaMemcpyFromSymbol(cells, c_cells, sizeof(cells));
    cudaMemcpyFromSymbol(bits, c_bits, sizeof(bits));
    cudaMemcpyFromSymbol(&scale, c_scale, sizeof(scale));
    cudaMemcpyFromSymbol(dir, c_dir, sizeof(dir));
    cudaMemcpyFromSymbol(&offset, c_offset, sizeof(offset));
    cudaMemcpyFromSymbol(&mask, c_mask, sizeof(mask));
    for (int i = 0; i < N; ++i) {
        const unsigned int u = static_cast<unsigned int>(i);
        y[i] = palette[u & 3u][0] * scale + dir[1] + axes[u & 1u][1] + lut[u & 255u] + static_cast<float>(cells[u % 3u][0]);
        z[i] = offset + cells[u % 3u][1];
        w[i] = (bits[u & 1u][2] + u) & mask;
    }
}

static void runAndCompare(int line) {
    int* z;
    float* y;
    unsigned int* w;
    cudaMalloc(reinterpret_cast<void**>(&y), N * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&z), N * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&w), N * sizeof(unsigned int));
    use_constants<<<1, 64>>>(y, z, w, N);
    std::vector<float> gy(N), ey(N);
    std::vector<int> gz(N), ez(N);
    std::vector<unsigned int> gw(N), ew(N);
    cudaMemcpy(gy.data(), y, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gz.data(), z, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gw.data(), w, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    expected(ey, ez, ew);
    bool ok = true;
    for (int i = 0; i < N; ++i) ok = ok && gy[i] == ey[i] && gz[i] == ez[i] && gw[i] == ew[i];
    check(ok, line);
    cudaFree(y);
    cudaFree(z);
    cudaFree(w);
}

int main() {
    cudagfx::setExecutionMode(cudagfx::ExecutionMode::CPU);
    runAndCompare(__LINE__); // initial values (the host initialisers, zeros for the rest)

    float scale = 5.0f;
    float palette[16];
    float lut[256];
    for (int i = 0; i < 16; ++i) palette[i] = 0.5f * static_cast<float>(i);
    for (int i = 0; i < 256; ++i) lut[i] = static_cast<float>(i) * 0.25f;
    cudaMemcpyToSymbol(c_scale, &scale, sizeof(scale));
    cudaMemcpyToSymbol(c_palette, palette, sizeof(palette));
    cudaMemcpyToSymbol(c_lut, lut, sizeof(lut));
    int cells[6] = {1, 10, 2, 20, 3, 30};
    cudaMemcpyToSymbol(c_cells, cells, sizeof(cells));
    unsigned int bits[8] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
    cudaMemcpyToSymbol(c_bits, bits, sizeof(bits));
    float axes[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}; // two float3: a 12-byte stride
    cudaMemcpyToSymbol(c_axes, axes, sizeof(axes));
    runAndCompare(__LINE__); // after the first updates

    // Partial update through an offset: only the second float2 component of c_dir, and one cell of c_cells.
    float dirY = 0.75f;
    cudaMemcpyToSymbol(c_dir, &dirY, sizeof(dirY), sizeof(float));
    int cell = 99;
    cudaMemcpyToSymbol(c_cells, &cell, sizeof(cell), 2 * sizeof(int)); // c_cells[1].x
    int offset = 1000;
    unsigned int mask = 0x0Fu;
    cudaMemcpyToSymbol(c_offset, &offset, sizeof(offset));
    cudaMemcpyToSymbol(c_mask, &mask, sizeof(mask));
    runAndCompare(__LINE__);

    // Read back what was written.
    float readBack[4];
    cudaMemcpyFromSymbol(readBack, c_palette, sizeof(readBack), 4 * sizeof(float));
    CHECK(readBack[0] == 2.0f && readBack[3] == 3.5f);
    int offsetBack = 0;
    cudaMemcpyFromSymbol(&offsetBack, c_offset, sizeof(offsetBack));
    CHECK(offsetBack == 1000);

    if (cudaGetLastError() != cudaSuccess) {
        std::printf("runtime error: %s\n", cudagfx::lastErrorMessage());
        return 2;
    }
    std::printf(g_bad ? "FAILED\n" : "OK\n");
    return g_bad ? 1 : 0;
}

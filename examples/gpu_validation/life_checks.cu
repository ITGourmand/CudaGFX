// Conway's Game of Life on a torus: two kernels (global memory, and a tile in shared memory with its halo, which
// needs a cooperative load and a barrier), ping-pong buffers, and a population count with atomics.
//
// The generations are chained on the device WITHOUT reading anything back in between: a launch must see what the
// previous one wrote (this is the ordering of bgfx::update / dispatch that has never run on a device). Integer
// logic only, so every result is compared bit for bit with a host simulation.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>

#include "gpu_checks.h"
#include "gv_common.h"

#define LIFE_TILE 16

__global__ void life_step(const unsigned int* cur, unsigned int* next, int w, int h) {
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
    unsigned int count = cur[ym * uw + xm] + cur[ym * uw + ux] + cur[ym * uw + xp] + cur[uy * uw + xm] + cur[uy * uw + xp] +
                         cur[yp * uw + xm] + cur[yp * uw + ux] + cur[yp * uw + xp];
    unsigned int alive = cur[uy * uw + ux];
    next[uy * uw + ux] = (count == 3u || (alive != 0u && count == 2u)) ? 1u : 0u;
}

__global__ void life_step_tiled(const unsigned int* cur, unsigned int* next, int w, int h) {
    __shared__ unsigned int tile[(LIFE_TILE + 2) * (LIFE_TILE + 2)];
    unsigned int uw = (unsigned int)w;
    unsigned int uh = (unsigned int)h;
    unsigned int lid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int originX = blockIdx.x * blockDim.x;
    unsigned int originY = blockIdx.y * blockDim.y;
    // Cooperative load of the (TILE + 2)^2 cells of the tile with its halo; wrapping handles the torus (and the
    // blocks that overhang the grid).
    for (unsigned int idx = lid; idx < (unsigned int)((LIFE_TILE + 2) * (LIFE_TILE + 2)); idx += blockDim.x * blockDim.y) {
        unsigned int ly = idx / (unsigned int)(LIFE_TILE + 2);
        unsigned int lx = idx % (unsigned int)(LIFE_TILE + 2);
        unsigned int gx = (originX + lx + uw - 1u) % uw;
        unsigned int gy = (originY + ly + uh - 1u) % uh;
        tile[idx] = cur[gy * uw + gx];
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return; // after the last barrier
    unsigned int cx = threadIdx.x + 1u;
    unsigned int cy = threadIdx.y + 1u;
    unsigned int stride = (unsigned int)(LIFE_TILE + 2);
    unsigned int count = tile[(cy - 1u) * stride + cx - 1u] + tile[(cy - 1u) * stride + cx] + tile[(cy - 1u) * stride + cx + 1u] +
                         tile[cy * stride + cx - 1u] + tile[cy * stride + cx + 1u] + tile[(cy + 1u) * stride + cx - 1u] +
                         tile[(cy + 1u) * stride + cx] + tile[(cy + 1u) * stride + cx + 1u];
    unsigned int alive = tile[cy * stride + cx];
    next[(unsigned int)y * uw + (unsigned int)x] = (count == 3u || (alive != 0u && count == 2u)) ? 1u : 0u;
}

__global__ void life_count(const unsigned int* cells, unsigned int* total, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (cells[i] != 0u) atomicAdd(&total[0], 1u);
}

namespace {

typedef std::vector<unsigned int> Grid;

void hostStep(const Grid& cur, Grid& next, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned int count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    count += cur[((y + dy + h) % h) * w + ((x + dx + w) % w)];
                }
            }
            const unsigned int alive = cur[y * w + x];
            next[y * w + x] = (count == 3u || (alive != 0u && count == 2u)) ? 1u : 0u;
        }
    }
}

Grid hostRun(Grid grid, int w, int h, int generations) {
    Grid next(grid.size());
    for (int g = 0; g < generations; ++g) {
        hostStep(grid, next, w, h);
        grid.swap(next);
    }
    return grid;
}

unsigned int population(const Grid& g) {
    unsigned int n = 0;
    for (unsigned int c : g) n += c;
    return n;
}

Grid randomSoup(int w, int h, float density, uint32_t seed) {
    Grid g(static_cast<size_t>(w) * h);
    uint32_t state = seed;
    for (auto& c : g) c = gv::random01(state) < density ? 1u : 0u;
    return g;
}

void place(Grid& g, int w, int x, int y, const char* const rows[], int count) {
    for (int r = 0; r < count; ++r) {
        for (int c = 0; rows[r][c] != '\0'; ++c) {
            if (rows[r][c] == '#') g[(y + r) * w + (x + c)] = 1u;
        }
    }
}

// `generations` steps chained on the device (ping-pong buffers, no readback in between), then a population count.
void runOnDevice(bool tiled, const Grid& initial, int w, int h, int generations, Grid& result, unsigned int& counted) {
    unsigned int* a = gv::upload(initial);
    unsigned int* b = gv::upload(Grid(initial.size(), 0u));
    std::vector<unsigned int> zero(1, 0u);
    unsigned int* dTotal = gv::upload(zero);
    dim3 grid((w + LIFE_TILE - 1) / LIFE_TILE, (h + LIFE_TILE - 1) / LIFE_TILE);
    for (int g = 0; g < generations; ++g) {
        if (tiled) life_step_tiled<<<grid, dim3(LIFE_TILE, LIFE_TILE)>>>(a, b, w, h);
        else life_step<<<grid, dim3(LIFE_TILE, LIFE_TILE)>>>(a, b, w, h);
        unsigned int* t = a;
        a = b;
        b = t;
    }
    const int cells = w * h;
    life_count<<<(cells + 255) / 256, 256>>>(a, dTotal, cells);
    result.assign(initial.size(), 0u);
    gv::download(result, a);
    gv::download(zero, dTotal);
    counted = zero[0];
    cudaFree(a);
    cudaFree(b);
    cudaFree(dTotal);
}

bool compare(const char* what, const Grid& got, const Grid& expected, int w) {
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            std::printf("    %s: cell (%d, %d) is %u, expected %u\n", what, static_cast<int>(i % w), static_cast<int>(i / w), got[i], expected[i]);
            return false;
        }
    }
    return true;
}

bool checkSoup(bool tiled, int w, int h, int generations, uint32_t seed) {
    const Grid initial = randomSoup(w, h, 0.3f, seed);
    const Grid expected = hostRun(initial, w, h, generations);
    Grid got;
    unsigned int counted = 0;
    runOnDevice(tiled, initial, w, h, generations, got, counted);
    if (!compare("soup", got, expected, w)) return false;
    if (counted != population(expected)) {
        std::printf("    population: counted %u, expected %u\n", counted, population(expected));
        return false;
    }
    return true;
}

bool checkSoupGlobal() { return checkSoup(false, 128, 128, 100, 12345u); }
bool checkSoupTiled() { return checkSoup(true, 96, 64, 100, 777u); }
bool checkSoupGlobalOddSize() { return checkSoup(false, 100, 70, 40, 99u); } // not a multiple of the block
bool checkSoupTiledOddSize() { return checkSoup(true, 100, 70, 40, 4242u); }

bool checkGlider() {
    const int w = 32, h = 32;
    const char* const glider[] = {".#.", "..#", "###"};
    Grid initial(w * h, 0u), expected(w * h, 0u);
    place(initial, w, 4, 4, glider, 3);
    place(expected, w, 14, 14, glider, 3); // 40 generations: ten cells down and to the right
    for (int tiled = 0; tiled < 2; ++tiled) {
        Grid got;
        unsigned int counted = 0;
        runOnDevice(tiled != 0, initial, w, h, 40, got, counted);
        if (!compare(tiled ? "glider (tiled)" : "glider", got, expected, w) || counted != 5u) return false;
    }
    return true;
}

bool checkOscillators() {
    const int w = 32, h = 32;
    const char* const blinker[] = {"###"};
    const char* const block[] = {"##", "##"};
    const char* const beehive[] = {".##.", "#..#", ".##."};
    Grid initial(w * h, 0u);
    place(initial, w, 3, 3, blinker, 1);
    place(initial, w, 10, 10, block, 2);
    place(initial, w, 20, 5, beehive, 3);
    for (int tiled = 0; tiled < 2; ++tiled) {
        Grid got;
        unsigned int counted = 0;
        runOnDevice(tiled != 0, initial, w, h, 2, got, counted); // period 2: the pattern is back
        if (!compare("oscillators after 2 generations", got, initial, w) || counted != 13u) return false;
        runOnDevice(tiled != 0, initial, w, h, 1, got, counted);
        if (!compare("oscillators after 1 generation", got, hostRun(initial, w, h, 1), w) || counted != 13u) return false;
    }
    return true;
}

} // namespace

int runLifeChecks(bool useGpu) {
    const gv::Check checks[] = {{"life: glider travels 10 cells (40 generations)", checkGlider},
                                {"life: still lifes and a blinker", checkOscillators},
                                {"life: 128x128 soup, 100 generations", checkSoupGlobal},
                                {"life: 96x64 soup, shared-memory tile", checkSoupTiled},
                                {"life: 100x70 soup (size not a block multiple)", checkSoupGlobalOddSize},
                                {"life: 100x70 soup, shared-memory tile", checkSoupTiledOddSize}};
    return gv::runChecks("Conway's Game of Life", checks, 6, useGpu);
}

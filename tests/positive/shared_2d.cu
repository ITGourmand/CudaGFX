// 2D shared array (tiled transpose), 2D blocks.
#define TILE 16
__global__ void shared_2d(float* out, const float* in, int w, int h) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < w && y < h) tile[threadIdx.y][threadIdx.x] = in[y * w + x];
    __syncthreads();
    int ox = blockIdx.y * TILE + threadIdx.x;
    int oy = blockIdx.x * TILE + threadIdx.y;
    if (ox < h && oy < w) out[oy * h + ox] = tile[threadIdx.x][threadIdx.y];
}

void host(float* out, float* in, int w, int h) {
    dim3 block(TILE, TILE);
    dim3 grid((w + TILE - 1) / TILE, (h + TILE - 1) / TILE);
    shared_2d<<<grid, block>>>(out, in, w, h);
}

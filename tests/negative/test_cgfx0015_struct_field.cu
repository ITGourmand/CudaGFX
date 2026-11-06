struct Cell { unsigned count; unsigned pad; float2 pos; };
__global__ void k(Cell* cells, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&cells[i].count, 1u);
}
// EXPECT error CGFX0015 4:26

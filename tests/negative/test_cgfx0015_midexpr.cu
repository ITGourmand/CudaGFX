__global__ void k(int* y) {
    y[0] = 2 * atomicAdd(&y[1], 1);
}
// EXPECT error CGFX0015 2:16

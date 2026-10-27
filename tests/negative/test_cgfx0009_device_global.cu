__device__ float g_counter = 0.0f;
__global__ void k(float* y) {
    y[0] = g_counter;
}
// EXPECT error CGFX0009 1:18

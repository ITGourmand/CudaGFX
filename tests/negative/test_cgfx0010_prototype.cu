__device__ float helper(float v);
__global__ void k(float* y) {
    y[0] = helper(1.0f);
}
// EXPECT error CGFX0010 3:12

__global__ void k(float* y) {
    size_t i = threadIdx.x;
    y[i] = 1.0f;
}
// EXPECT warning CGFX0024 2:12

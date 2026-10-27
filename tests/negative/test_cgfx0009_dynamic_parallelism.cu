__global__ void inner(float* y) { y[0] = 1.0f; }
__global__ void outer(float* y) {
    inner<<<1, 32>>>(y);
}
// EXPECT error CGFX0009 3:5

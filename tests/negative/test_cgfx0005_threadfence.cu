__global__ void k(float* y) {
    y[0] = 1.0f;
    __threadfence();
}
// EXPECT error CGFX0005 3:5

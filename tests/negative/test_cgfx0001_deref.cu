__global__ void k(float* y) {
    *y = 1.0f;
}
// EXPECT error CGFX0001 2:5

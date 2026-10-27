__global__ void k(float* y, float a, float b) {
    y[0] = a % b;
}
// EXPECT error CGFX0019 2:14

__global__ void k(float* y, int n) {
    int j = (int64_t)n;
    y[0] = (float)j;
}
// EXPECT error CGFX0002 2:13

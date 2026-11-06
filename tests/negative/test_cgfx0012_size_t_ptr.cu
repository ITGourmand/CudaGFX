__global__ void k(const size_t* a, float* y) {
    y[0] = (float)a[0];
}
// EXPECT error CGFX0012 1:33

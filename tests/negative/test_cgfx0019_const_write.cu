__global__ void k(const float* c, float* y) {
    c[0] = 1.0f;
    y[0] = c[0];
}
// EXPECT error CGFX0019 2:5

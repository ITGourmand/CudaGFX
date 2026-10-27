__global__ void k(float* y, float a, float b) {
    y[0] = fmodf(a, b);
}
// EXPECT warning CGFX0020 2:12

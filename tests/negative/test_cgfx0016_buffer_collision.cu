__global__ void k(float* x) {
    float xBuffer = 1.0f;
    x[0] = xBuffer;
}
// EXPECT error CGFX0016 1:26

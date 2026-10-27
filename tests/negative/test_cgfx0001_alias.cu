__global__ void k(float* y, float* z) {
    float* q = y;
    q[0] = z[0];
}
// EXPECT error CGFX0001 2:12

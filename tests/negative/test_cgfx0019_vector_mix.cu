__global__ void k(float2* y, float2 a, int4 b) {
    y[0] = a + b;
}
// EXPECT error CGFX0019 2:14

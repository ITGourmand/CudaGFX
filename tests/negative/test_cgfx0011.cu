__global__ void k1(float* p) { p[0] = 1.0f; }
__global__ void k2(float* p) { p[0] = 2.0f; }
// EXPECT error CGFX0011 2:17

__global__ void k(float* y) { y[0] = 1.0f; }
#define GO(p) k<<<1, 64>>>(p)
// EXPECT error CGFX0013 2:16

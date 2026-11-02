__constant__ float table[];
__global__ void k(float* y) { y[0] = 1.0f; }
// EXPECT error CGFX0009 1:20

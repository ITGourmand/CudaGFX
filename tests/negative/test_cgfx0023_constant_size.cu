__constant__ float table[4097]; // 16 388 bytes: too big for a uniform array, and no initializer to bake
__global__ void k(float* y) { y[0] = table[3]; }
// EXPECT error CGFX0023 2:38

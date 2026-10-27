__global__ void k(float* a) {
    float x = __shfl_sync(0xFFFFFFFF, a[0], 0);
}
// EXPECT error CGFX0003 2:15

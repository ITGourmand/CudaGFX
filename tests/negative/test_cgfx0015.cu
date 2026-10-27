__global__ void k_atomic(float* p) {
    atomicAdd(&p[0], 1.0f);
}
// EXPECT error CGFX0015 2:15

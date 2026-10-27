__global__ void k(float* y) {
    __shared__ float big[10000];
    big[threadIdx.x] = 1.0f;
    y[0] = big[0];
}
// EXPECT error CGFX0008 1:17

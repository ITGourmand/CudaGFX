__global__ void k(float* y) {
    extern __shared__ float s[];
    s[threadIdx.x] = y[threadIdx.x];
}
void host(float* d) {
    k<<<1, 64, 64 * sizeof(float)>>>(d);
}
// EXPECT error CGFX0004 6:16

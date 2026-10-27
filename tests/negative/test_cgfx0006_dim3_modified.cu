__global__ void k(float* y) { y[0] = 1.0f; }
void host(float* d) {
    dim3 b(32);
    b.x = 64;
    k<<<1, b>>>(d);
}
// EXPECT error CGFX0006 5:12

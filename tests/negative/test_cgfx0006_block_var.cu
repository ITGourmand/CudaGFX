__global__ void k(float* y) { y[0] = 1.0f; }
void host(float* d) {
    int b = 64;
    k<<<1, b>>>(d);
}
// EXPECT error CGFX0006 4:12

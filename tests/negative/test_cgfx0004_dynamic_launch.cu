__global__ void k(float* y) { y[0] = 1.0f; }
void host(float* d, int bytes) {
    k<<<1, 64, bytes>>>(d);
}
// EXPECT error CGFX0004 3:16

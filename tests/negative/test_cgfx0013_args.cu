__global__ void k(float* y) { y[0] = 1.0f; }
void host(float* d) {
    k<<<1>>>(d);
}
// EXPECT error CGFX0013 3:5

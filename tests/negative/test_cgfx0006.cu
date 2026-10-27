__global__ void k_dummy(float* p) { p[0] = 1.0f; }
void host_fn(int n, float* p) {
    k_dummy<<<1, n>>>(p);
}
// EXPECT error CGFX0006 3:18

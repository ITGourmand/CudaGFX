__global__ void k_dummy(float* p) { p[0] = 1.0f; }
void host_fn(float* p) {
    k_dummy<<<1, 256(p);
}
// EXPECT error CGFX0013 3:5

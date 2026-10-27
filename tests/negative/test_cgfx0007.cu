__launch_bounds__(256)
__global__ void k_bounds(float* p) { p[0] = 1.0f; }
void host_fn(float* p) {
    k_bounds<<<1, 128>>>(p);
}
// EXPECT warning CGFX0007 4:5

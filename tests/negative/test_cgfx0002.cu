__global__ void k_double(double x) {
    (void)x;
}
// EXPECT error CGFX0002 1:33

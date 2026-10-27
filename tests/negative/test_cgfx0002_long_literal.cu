__global__ void k(int* y) {
    y[0] = 5000000000;
}
// EXPECT error CGFX0002 2:12

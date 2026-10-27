typedef float real;
__global__ void k(real* y) {
    y[0] = 1.0f;
}
// EXPECT error CGFX0010 2:25

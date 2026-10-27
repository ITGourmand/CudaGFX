__global__ void k(float* y) {
    y[0] = no_such_function(2.0f);
}
// EXPECT error CGFX0010 2:12

__global__ void k(float* y) {
    float x = 1.0f;
    x = x + undeclared_var;
    y[0] = x;
}
// EXPECT error CGFX0010 3:13

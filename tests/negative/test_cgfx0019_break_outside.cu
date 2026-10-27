__global__ void k(float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = 1.0f;
    }
    break;
}
// EXPECT error CGFX0019 5:5

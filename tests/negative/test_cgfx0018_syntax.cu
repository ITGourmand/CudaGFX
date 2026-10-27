__global__ void k(float* p) {
    p[0] = 1.0f
    p[1] = 2.0f;
}
// EXPECT error CGFX0018 3:5

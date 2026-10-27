__global__ void k(float* a) {
    printf("in unsupported\n");
}
// EXPECT error CGFX0005 2:5

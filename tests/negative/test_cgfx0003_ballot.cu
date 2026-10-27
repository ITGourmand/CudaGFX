__global__ void k(int* y) {
    y[0] = __ballot_sync(0xFFFFFFFF, 1);
}
// EXPECT error CGFX0003 2:12

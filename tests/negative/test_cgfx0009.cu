__global__ void k_virtual() {
    throw 42;
}
// EXPECT error CGFX0009 2:5

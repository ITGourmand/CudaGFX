template <typename T>
__global__ void k(T* y) {
    y[0] = 1;
}
// EXPECT error CGFX0009 1:1

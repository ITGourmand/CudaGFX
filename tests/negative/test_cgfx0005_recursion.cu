__device__ int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }
__global__ void k(int* y) {
    y[0] = fact(5);
}
// EXPECT error CGFX0005 1:54

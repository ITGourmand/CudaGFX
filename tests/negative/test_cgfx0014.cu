#define CONCAT(a, b) a ## b
__global__ void k_macro(int* y) { y[0] = CONCAT(1, 0); }
// EXPECT error CGFX0014 2:42

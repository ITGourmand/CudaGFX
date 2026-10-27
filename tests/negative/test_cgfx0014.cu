#define CONCAT(a, b) a ## b
__global__ void k_macro() {}
// EXPECT error CGFX0014 1:24

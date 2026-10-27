__global__ void k_signed_div(int* p, int a, int b) {
    p[0] = a % b;
}
// EXPECT warning CGFX0017 2:14

__global__ void k_ptr_arith(float* p) {
    p++;
}
// EXPECT error CGFX0001 2:5

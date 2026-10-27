__global__ void k_undeclared(float* p) {
    p[0] = undeclared_identifier;
}
// EXPECT error CGFX0010 2:12

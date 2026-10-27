struct Params { float a; float b; };
__global__ void k(float* y, Params p) {
    y[0] = p.a;
}
// EXPECT error CGFX0002 2:36

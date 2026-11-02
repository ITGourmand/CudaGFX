struct Params {
    float scale;
    float bias;
};
__constant__ Params params;
__global__ void k(float* y) { y[0] = params.scale; }
// EXPECT error CGFX0023 6:38

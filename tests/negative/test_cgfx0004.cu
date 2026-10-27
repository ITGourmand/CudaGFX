__global__ void k_extern_shared() {
    extern __shared__ float s[];
}
// EXPECT error CGFX0004 2:5

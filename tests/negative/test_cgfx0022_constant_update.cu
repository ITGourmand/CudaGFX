// A bool constant is baked into the shader (it has no uniform layout): updating it at run time would be ignored by the GPU.
__constant__ bool flags[2] = {true, false};
__global__ void k(float* y) { y[0] = flags[1] ? 1.0f : 0.0f; }
void host_update(const bool* h) {
    cudaMemcpyToSymbol(flags, h, 2 * sizeof(bool));
}
// EXPECT error CGFX0022 5:5

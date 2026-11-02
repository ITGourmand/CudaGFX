// __constant__ objects of every supported type: they reach the shader through uniform arrays that the runtime refills from
// the host object at each launch, so cudaMemcpyToSymbol is legal.
__constant__ float c_scale = 2.0f;
__constant__ int c_offset = 3;
__constant__ unsigned int c_mask = 0xFFu;
__constant__ float2 c_dir = {1.0f, 0.0f};
__constant__ float3 c_axes[2];
__constant__ float4 c_palette[4];
__constant__ int2 c_cells[3];
__constant__ uint4 c_bits[2];
__constant__ float c_lut[256];

__device__ float shade(unsigned int i) { return c_palette[i & 3u].x * c_scale + c_dir.y; }

__global__ void use_constants(float* y, int* z, unsigned int* w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = shade((unsigned int)i) + c_axes[i & 1].y + c_lut[(unsigned int)i & 255u] + (float)c_cells[i % 3u].x;
    z[i] = c_offset + c_cells[(unsigned int)i % 3u].y;
    w[i] = (c_bits[(unsigned int)i & 1u].z + (unsigned int)i) & c_mask;
}

void update_constants(float scale, const float* palette, const float* lut) {
    cudaMemcpyToSymbol(c_scale, &scale, sizeof(scale));
    cudaMemcpyToSymbol(c_palette, palette, 4 * sizeof(float4));
    cudaMemcpyToSymbol(c_lut, lut, 256 * sizeof(float));
}

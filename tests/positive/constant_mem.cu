// Constant memory declaration and access in kernel
__constant__ float c_weights[4] = {1.0f, 2.0f, 3.0f, 4.0f};
__constant__ float c_scale = 0.5f;

__global__ void constant_mem_kernel(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned u = (unsigned)i;
        float w = c_weights[u % 4u];
        out[i] = in[i] * w * c_scale;
    }
}

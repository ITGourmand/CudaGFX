// gridDim: declared as a uniform and filled in by the runtime.
__global__ void grid_dim(float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) y[i] = float(gridDim.x);
}

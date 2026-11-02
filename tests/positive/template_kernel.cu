// Instantiated kernel template with integer parameter and __shared__ memory
template <int BLOCK_SIZE>
__global__ void reduce_temp(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    if (BLOCK_SIZE >= 256 && tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

void launch_helper(float* d_in, float* d_out, int n) {
    reduce_temp<256><<<1, 256>>>(d_in, d_out, n);
}

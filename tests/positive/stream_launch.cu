// Streams and events compile and run serially (bgfx has a single in-order queue).
__global__ void stream_launch(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i];
}

int main() {
    float *a = nullptr, *out = nullptr;
    cudaMalloc((void**)&a, 1024 * sizeof(float));
    cudaMalloc((void**)&out, 1024 * sizeof(float));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    stream_launch<<<4, 256, 0, stream>>>(a, out, 1024);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return 0;
}

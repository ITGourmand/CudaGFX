// size_t, ptrdiff_t: values become 32-bit integers (warning CGFX0024); allocations are capped at 2^31 - 1 bytes.
__device__ size_t twice(size_t v) {
    return v * 2;
}

__global__ void size_index(const float* a, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    ptrdiff_t back = -1;
    if (i < n) out[i] = a[i] + (float)back + (float)twice(i);
}

int main() {
    float *a = nullptr, *out = nullptr;
    const size_t n = 1024;
    cudaMalloc((void**)&a, n * sizeof(float));
    cudaMalloc((void**)&out, n * sizeof(float));
    size_index<<<4, 256>>>(a, out, n);
    return 0;
}

// (size_t)n: the cast fix of J0 and the 32-bit size_t of J1 together.
__global__ void size_cast(const float* a, float* out, int n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)n) out[i] = a[i];
}

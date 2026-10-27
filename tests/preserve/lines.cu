#include <cuda_runtime.h> // L1
// L2
__global__ void scale(float* y, float a, int n) { // L3
    int i = blockIdx.x * blockDim.x + threadIdx.x; // L4
    if (i < n) y[i] *= a; // L5
} // L6
// L7
__device__ float twice(float v) { return 2.f * v; } // L8
// L9
int main() { // L10
    float* d = nullptr; // L11
    cudaMalloc((void**)&d, 1024); // L12
    scale<<<4, // L13
            256>>>(d, // L14
                   2.f, 1024); // L15
    int after = 1; // L16
    cudaFree(d); // L17
    return after; // L18
} // L19

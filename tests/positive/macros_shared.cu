// Object-like and function-like macros, shared array sized by a macro and by a global constant.
#include <cuda_runtime.h>
#define BS 128
#define SQ(x) ((x) * (x))
const int HALF = BS / 2;

__global__ void macros_shared(float* y, int n) {
    __shared__ float t[BS];
    __shared__ float u[HALF + 1];
    int i = threadIdx.x;
    t[i] = y[i];
    u[i % HALF] = SQ(t[i]);
    __syncthreads();
    y[i] = t[BS - 1 - i] + u[i % HALF];
}

void host(float* y, int n) {
    macros_shared<<<1, BS>>>(y, n);
}

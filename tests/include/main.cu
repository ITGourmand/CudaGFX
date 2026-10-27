#include <cuda_runtime.h>
#include "helper.cuh"

__global__ void offset_scale(float* y) {
    y[threadIdx.x] = scaled(y[threadIdx.x]) + OFFSET;
}

// Host code keeps its macros, # and ## included: the C++ compiler expands them, not CudaGFX.
#include <cstdio>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { std::printf("%s failed: %s\n", #call, cudaGetErrorString(e)); } } while (0)
#define GLUE(a, b) a##b

__global__ void macro_check_host(float* y) {
    y[threadIdx.x] = 1.0f;
}

int main() {
    float* GLUE(device, Buffer) = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&deviceBuffer, 256 * sizeof(float)));
    macro_check_host<<<1, 256>>>(deviceBuffer);
    CUDA_CHECK(cudaFree(deviceBuffer));
    return 0;
}

// CPU backend, kernel with barriers: a block reduction over several blocks, the last one partial, whose threads
// past the end of the data return BEFORE the __syncthreads() calls (this used to deadlock the CPU executor).
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define BS 128

__global__ void reduce_sum(const float* in, float* out, int n) {
    __shared__ float sdata[BS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sdata[tid] = in[i];
    __syncthreads();
    for (int s = BS / 2; s > 0; s = s / 2) {
        if (tid < s && i + s < n) sdata[tid] = sdata[tid] + sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main() {
    cudagfx::setExecutionMode(cudagfx::ExecutionMode::CPU);
    const int n = BS * 5 + 77;
    const int blocks = (n + BS - 1) / BS;
    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) in[i] = static_cast<float>(i % 7);
    std::vector<float> out(blocks, -1.f);

    float *dIn = nullptr, *dOut = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dIn), n * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dOut), blocks * sizeof(float));
    cudaMemcpy(dIn, in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    reduce_sum<<<blocks, BS>>>(dIn, dOut, n);
    cudaMemcpy(out.data(), dOut, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int b = 0; b < blocks; ++b) {
        float expected = 0.f;
        for (int i = b * BS; i < n && i < (b + 1) * BS; ++i) expected += in[i];
        if (out[b] != expected) {
            std::printf("block %d: got %f, expected %f\n", b, out[b], expected);
            ++bad;
        }
    }
    cudaFree(dIn);
    cudaFree(dOut);
    if (cudaGetLastError() != cudaSuccess) return 2;
    std::printf(bad ? "FAILED\n" : "OK\n");
    return bad ? 1 : 0;
}

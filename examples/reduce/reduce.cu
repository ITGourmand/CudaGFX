#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

__global__ void reduce_sum(const float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (int s = 128; s > 0; s = s / 2) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int n = 1024;
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    std::vector<float> h_in(n, 1.0f);
    std::vector<float> h_out(numBlocks, 0.0f);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_in, n * sizeof(float));
    cudaMalloc((void**)&d_out, numBlocks * sizeof(float));

    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum<<<numBlocks, 256>>>(d_in, d_out, n);

    cudaMemcpy(h_out.data(), d_out, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float totalGpu = 0.0f;
    for (int b = 0; b < numBlocks; ++b) {
        totalGpu += h_out[b];
    }

    float totalCpu = 0.0f;
    for (int i = 0; i < n; ++i) {
        totalCpu += h_in[i];
    }

    std::printf("Reduce GPU Sum = %f | CPU Sum = %f\n", totalGpu, totalCpu);

    cudaFree(d_in);
    cudaFree(d_out);

    if (std::fabs(totalGpu - totalCpu) > 1e-4f) {
        std::printf("VERIFICATION FAILED!\n");
        return 1;
    }
    std::printf("VERIFICATION PASSED!\n");
    return 0;
}

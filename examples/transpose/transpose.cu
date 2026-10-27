#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

__device__ int index2D(int x, int y, int stride) {
    return y * stride + x;
}

__global__ void transpose2d(const float* src, float* dst, int width, int height) {
    __shared__ float tile[256];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y * 16 + threadIdx.x] = src[index2D(x, y, width)];
    }
    __syncthreads();

    int outX = blockIdx.y * blockDim.y + threadIdx.x;
    int outY = blockIdx.x * blockDim.x + threadIdx.y;

    if (outX < height && outY < width) {
        dst[index2D(outX, outY, height)] = tile[threadIdx.x * 16 + threadIdx.y];
    }
}

int main() {
    const int width = 64;
    const int height = 64;
    const int total = width * height;

    std::vector<float> h_in(total);
    std::vector<float> h_out(total, 0.0f);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_in[y * width + x] = static_cast<float>(y * 100 + x);
        }
    }

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_in, total * sizeof(float));
    cudaMalloc((void**)&d_out, total * sizeof(float));

    cudaMemcpy(d_in, h_in.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    transpose2d<<<grid, block>>>(d_in, d_out, width, height);

    cudaMemcpy(h_out.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float expected = h_in[y * width + x];
            float actual = h_out[x * height + y];
            if (std::fabs(actual - expected) > 1e-4f) {
                ok = false;
                break;
            }
        }
    }

    std::printf("Transpose 2D Verification: %s\n", ok ? "PASSED" : "FAILED");

    cudaFree(d_in);
    cudaFree(d_out);
    return ok ? 0 : 1;
}

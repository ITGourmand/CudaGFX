#include "saxpy_gen_kernels.h"
#include <vector>
#include <cstdio>








int main()
{
    const int n = 1 << 20;
    std::vector<float> hx(n, 1.0f), hy(n, 2.0f);

    float *dx = nullptr, *dy = nullptr;
    cudaMalloc((void**)&dx, n * sizeof(float));
    cudaMalloc((void**)&dy, n * sizeof(float));
    cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cudagfx_gen::saxpy::launch((n + 255) / 256, 256, 2.0f, dx, dy, n);

    cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::printf("y[0] = %f (attendu 4.0)\n", hy[0]);

    cudaFree(dx);
    cudaFree(dy);
    return 0;
}

// __device__ functions: return types, prototypes, call chains, min/max on integers.
__device__ int twice(int v);
__device__ unsigned pick(unsigned a, unsigned b) { return min(a, b); }
__device__ __forceinline__ float sq(float v) { return v * v; }
__device__ float norm2(float3 p) { return sq(p.x) + sq(p.y) + sq(p.z); }
__device__ int twice(int v) { return v * 2; }

__global__ void device_funcs(int* y, const float* __restrict__ x, int n) {
    int i = threadIdx.x;
    int m = max(twice(i), 3);
    y[i] = m + int(pick(1u, 2u)) + int(norm2(make_float3(x[i], 1.f, 2.f)));
}

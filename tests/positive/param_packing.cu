// Parameter packing: float2 on an even lane, float3, int4, bool, uint.
__global__ void param_packing(float* y, float a, float b, float2 v, float3 w, int4 q, bool flag, unsigned m, int n) {
    int i = threadIdx.x;
    if (flag) y[i] = a + b + v.x + v.y + w.z + float(q.w) + float(m) + float(n);
}

// Vector arithmetic (component-wise), scalar broadcast, components.
__global__ void vector_ops(float4* y, const float4* x, float4 a, float s, int n) {
    int i = threadIdx.x;
    float4 v = x[i];
    float4 r = v + a * s;
    r = r - v;
    r.x = r.y * 2.f;
    y[i] = r;
}

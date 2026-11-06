// fmaf, __fmaf_rn, __fdividef, __saturatef, copysignf, isfinite and __ldg.
__global__ void math_more(float* y, const float* __restrict__ x) {
    int i = threadIdx.x;
    float v = __ldg(&x[i]);
    float s = fmaf(v, 2.0f, 1.0f) + __fmaf_rn(v, v, 0.5f) + __fdividef(v, 3.0f);
    s += __saturatef(v) + copysignf(1.5f, v);
    if (!isfinite(s)) s = 0.0f;
    y[i] = s;
}

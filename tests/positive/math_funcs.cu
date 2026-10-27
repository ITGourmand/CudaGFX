// Native math functions and bit conversions.
__global__ void math_funcs(float* y, const float* x, int* z) {
    int i = threadIdx.x;
    float v = x[i];
    y[i] = sinf(v) + cosf(v) + sqrtf(fabsf(v)) + fminf(v, 1.f) + fmaxf(v, 0.f) + powf(v, 2.f)
         + rsqrtf(v + 2.f) + floorf(v) + atan2f(v, 1.f) + expf(v) + logf(v + 2.f);
    z[i] = __float_as_int(v) ^ (int)__popc(7u);
    if (isnan(v)) y[i] = 0.f;
}

// Conversions: (unsigned), float -> int, mixed ternary, negation, ~, !, mixed comparisons.
__global__ void conversions(float* y, int* z, unsigned u, int n) {
    int i = threadIdx.x;
    float f = (unsigned)i * 0.5f;
    int t = (i > 3) ? n : 7;
    unsigned w = u + (unsigned)(-n);
    bool lt = i < u;
    z[i] = t + (int)f + (int)(~w & 0xFFu) + (lt ? 1 : 0) + (!n ? 1 : 0);
    y[i] = f - (float)t;
}

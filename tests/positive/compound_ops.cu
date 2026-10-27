// Compound assignments, including with an implicit conversion (int += float).
__global__ void compound_ops(unsigned* y, float* f, int n) {
    int i = threadIdx.x;
    unsigned x = y[i];
    x |= n;
    x <<= 2;
    x %= 7u;
    int k = i;
    k += 1.5f;
    k *= 2.f;
    f[i] += float(k);
    f[i] /= 3;
    y[i] = x;
}

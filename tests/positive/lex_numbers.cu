// Literals: 1.f, 2., exponents, hexadecimal, u suffixes, digit separators.
__global__ void lex_numbers(float* y, unsigned* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = 1.f + 2. * 1e-3f + 0.5e1 + .25f;
        unsigned m = 0xFFu + 1'000u + 0b101u + 017u;
        y[i] = a / 2.f;
        z[i] = m << 1u;
    }
}

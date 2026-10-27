// Assorted cases: argument and return conversions, swizzles, nested indexing, loops, early return.
struct Cell { int id; float w[3]; };

__device__ float blend(float a, float b, float t) { return a + (b - a) * t; }
__device__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ unsigned hash32(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return x;
}
__device__ float4 shade(float4 c, float k) { return c * k; }

__global__ void misc_semantics(float4* img, const int* idx, const float* src, int* out, int w, int h, float t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int i = clampi(y * w + x, 0, w * h - 1);          // return int, int arguments
    float f = blend(src[idx[i]], 1, t);               // int -> float argument, nested indexing
    float4 c = img[i];
    c.xy = c.zw * 0.5f;                               // 2-component swizzle
    c.w = f > 0.5f ? 1 : 0;                           // mixed int/float ternary
    img[i] = shade(c, f);

    Cell cell;
    cell.id = int(hash32(unsigned(i)) & 7u);
    cell.w[1] = f;
    unsigned s = 0;
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            if (b == 2) continue;
            if (a == 2) break;
            s += unsigned(a * 3 + b);
        }
    }
    bool even = (s & 1u) == 0u;
    int sh = 1 << (cell.id + 1);
    out[i] = even ? sh : -sh;
    out[i] += (int)(cell.w[1] * 4.0f);
    while (true) { s >>= 1; if (s == 0u) break; }
}

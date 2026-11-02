struct Big {
    float4 a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16;
};
__global__ void k(Big* p) { p[0].a0.x = 1.0f; }
// EXPECT error CGFX0021 4:24

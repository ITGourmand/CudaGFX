struct Bad {
    float a;
    float4 b; // offset 16 on the host (16-byte aligned), 16 in std430, but 4 in HLSL structured buffers
};
__global__ void k(Bad* p) { p[0].a = 1.0f; }
// EXPECT error CGFX0021 5:24

struct Bad {
    float4 a;
    float b; // 20 bytes of data: 20 in HLSL, 32 in C++ and std430
};
__global__ void k(Bad* p) { p[0].b = 1.0f; }
// EXPECT error CGFX0021 5:24

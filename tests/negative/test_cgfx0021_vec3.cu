struct Bad {
    float3 v;
    float w;
};
__global__ void k(Bad* p) { p[0].w = 1.0f; }
// EXPECT error CGFX0021 5:24

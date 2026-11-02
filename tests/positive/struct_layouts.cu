// Structs used as buffer elements whose layout is the same on the host, in std430 and in HLSL: vectors at their
// natural alignment, explicit padding, nested structs.
struct Mixed {
    float4 color;
    float2 uv;
    float2 pad0;
};

struct Item {
    int id;
    float weight;
    float2 uv;
    float4 color;
};

struct Pair {
    Item first;
    Item second;
};

__global__ void shade(Mixed* m, const Item* items, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    m[i].color = items[i].color * items[i].weight;
    m[i].uv = items[i].uv;
    m[i].pad0 = make_float2(0.0f, 0.0f);
}

__global__ void swap_pairs(Pair* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Item t = p[i].first;
    p[i].first = p[i].second;
    p[i].second = t;
}

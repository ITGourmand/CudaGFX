// Local POD struct, global constant (const / constexpr).
const int N = 64;
constexpr float SCALE = 0.5f;
struct Particle { float x; float v; float2 pad; };

__global__ void structs_consts(float* y) {
    Particle p;
    p.x = 1.f;
    p.v = SCALE * N;
    p.pad = make_float2(0.f, 1.f);
    y[threadIdx.x] = p.x + p.v + p.pad.y;
}

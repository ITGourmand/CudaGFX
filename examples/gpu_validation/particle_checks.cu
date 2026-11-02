// Two particle systems, O(N^2) interactions on a torus, ping-pong struct buffers:
//
//   Particle Life   typed particles attracting / repelling each other through an attraction matrix kept in
//                   __constant__ memory, a 32-byte struct made of two float2 (8-byte aligned fields)
//   Primordial      Schmickl et al.'s Primordial Particle System: every particle turns according to the number of
//   Particle System neighbours on its left and right, and moves; trigonometry and discrete decisions
//
// Each step reads the previous state of ALL the particles, so the steps cannot be fused: the launches are chained on
// the device. Floating-point results are compared with a host simulation (tolerance, and a fraction of particles for
// the system with discrete decisions, where a rounding difference can flip one neighbour on one particle).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gpu_checks.h"
#include "gv_common.h"

struct Body {
    float2 pos;
    float2 vel;
    int type;
    int id;
    float2 pad;
};

struct Pps {
    float2 pos;
    float phi;
    int neighbors;
};

__constant__ float c_attraction[16] = {0.8f,  0.3f, -0.4f, 0.0f,  //
                                        -0.5f, 0.9f, 0.2f,  -0.3f, //
                                        0.3f,  -0.6f, 0.7f, 0.4f,  //
                                        0.1f,  0.5f, -0.2f, 0.6f};

__global__ void pl_step(const Body* in, Body* out, int n, float dt, float rmax, float beta, float friction, float forceScale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Body me = in[i];
    float fx = 0.0f;
    float fy = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        Body other = in[j];
        float dx = other.pos.x - me.pos.x;
        float dy = other.pos.y - me.pos.y;
        dx = dx - floorf(dx + 0.5f); // torus of size 1
        dy = dy - floorf(dy + 0.5f);
        float r = sqrtf(dx * dx + dy * dy);
        if (r > 0.0f && r < rmax) {
            float a = c_attraction[(unsigned int)me.type * 4u + (unsigned int)other.type];
            float f;
            if (r < beta * rmax) f = r / (beta * rmax) - 1.0f;
            else f = a * (1.0f - fabsf(1.0f + beta - 2.0f * r / rmax) / (1.0f - beta));
            fx = fx + dx / r * f;
            fy = fy + dy / r * f;
        }
    }
    float vx = me.vel.x * friction + fx * forceScale * dt;
    float vy = me.vel.y * friction + fy * forceScale * dt;
    float px = me.pos.x + vx * dt;
    float py = me.pos.y + vy * dt;
    px = px - floorf(px);
    py = py - floorf(py);
    Body o;
    o.pos = make_float2(px, py);
    o.vel = make_float2(vx, vy);
    o.type = me.type;
    o.id = me.id;
    o.pad = make_float2(0.0f, 0.0f);
    out[i] = o;
}

__global__ void pps_step(const Pps* in, Pps* out, int n, float size, float radius, float alpha, float beta, float speed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Pps me = in[i];
    float c = cosf(me.phi);
    float s = sinf(me.phi);
    float r2 = radius * radius;
    int total = 0;
    int left = 0;
    int right = 0;
    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        float dx = in[j].pos.x - me.pos.x;
        float dy = in[j].pos.y - me.pos.y;
        dx = dx - size * floorf(dx / size + 0.5f);
        dy = dy - size * floorf(dy / size + 0.5f);
        if (dx * dx + dy * dy < r2) {
            total = total + 1;
            float side = c * dy - s * dx; // > 0: the neighbour is on the left of the heading
            if (side > 0.0f) left = left + 1;
            else right = right + 1;
        }
    }
    float turn = alpha + beta * (float)total * ((right > left) ? 1.0f : ((right < left) ? -1.0f : 0.0f));
    float phi = me.phi + turn;
    float nx = me.pos.x + speed * cosf(phi);
    float ny = me.pos.y + speed * sinf(phi);
    nx = nx - size * floorf(nx / size);
    ny = ny - size * floorf(ny / size);
    Pps o;
    o.pos = make_float2(nx, ny);
    o.phi = phi;
    o.neighbors = total;
    out[i] = o;
}

namespace {

const float kAttraction[16] = {0.8f,  0.3f, -0.4f, 0.0f, -0.5f, 0.9f, 0.2f,  -0.3f, 0.3f, -0.6f, 0.7f, 0.4f, 0.1f, 0.5f, -0.2f, 0.6f};

// Same expressions, same order of operations as the kernel.
void hostPlStep(const std::vector<Body>& in, std::vector<Body>& out, float dt, float rmax, float beta, float friction, float forceScale) {
    const int n = static_cast<int>(in.size());
    for (int i = 0; i < n; ++i) {
        const Body me = in[i];
        float fx = 0.0f, fy = 0.0f;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const Body other = in[j];
            float dx = other.pos.x - me.pos.x;
            float dy = other.pos.y - me.pos.y;
            dx = dx - std::floor(dx + 0.5f);
            dy = dy - std::floor(dy + 0.5f);
            const float r = std::sqrt(dx * dx + dy * dy);
            if (r > 0.0f && r < rmax) {
                const float a = kAttraction[static_cast<unsigned>(me.type) * 4u + static_cast<unsigned>(other.type)];
                float f;
                if (r < beta * rmax) f = r / (beta * rmax) - 1.0f;
                else f = a * (1.0f - std::fabs(1.0f + beta - 2.0f * r / rmax) / (1.0f - beta));
                fx = fx + dx / r * f;
                fy = fy + dy / r * f;
            }
        }
        const float vx = me.vel.x * friction + fx * forceScale * dt;
        const float vy = me.vel.y * friction + fy * forceScale * dt;
        float px = me.pos.x + vx * dt;
        float py = me.pos.y + vy * dt;
        px = px - std::floor(px);
        py = py - std::floor(py);
        out[i].pos = make_float2(px, py);
        out[i].vel = make_float2(vx, vy);
        out[i].type = me.type;
        out[i].id = me.id;
        out[i].pad = make_float2(0.0f, 0.0f);
    }
}

void hostPpsStep(const std::vector<Pps>& in, std::vector<Pps>& out, float size, float radius, float alpha, float beta, float speed) {
    const int n = static_cast<int>(in.size());
    for (int i = 0; i < n; ++i) {
        const Pps me = in[i];
        const float c = std::cos(me.phi);
        const float s = std::sin(me.phi);
        const float r2 = radius * radius;
        int total = 0, left = 0, right = 0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            float dx = in[j].pos.x - me.pos.x;
            float dy = in[j].pos.y - me.pos.y;
            dx = dx - size * std::floor(dx / size + 0.5f);
            dy = dy - size * std::floor(dy / size + 0.5f);
            if (dx * dx + dy * dy < r2) {
                total = total + 1;
                const float side = c * dy - s * dx;
                if (side > 0.0f) left = left + 1;
                else right = right + 1;
            }
        }
        const float turn = alpha + beta * static_cast<float>(total) * ((right > left) ? 1.0f : ((right < left) ? -1.0f : 0.0f));
        const float phi = me.phi + turn;
        float nx = me.pos.x + speed * std::cos(phi);
        float ny = me.pos.y + speed * std::sin(phi);
        nx = nx - size * std::floor(nx / size);
        ny = ny - size * std::floor(ny / size);
        out[i].pos = make_float2(nx, ny);
        out[i].phi = phi;
        out[i].neighbors = total;
    }
}

// Distance between two coordinates of a torus of the given size.
float torusDistance(float a, float b, float size) {
    float d = std::fabs(a - b);
    return d > size * 0.5f ? size - d : d;
}

bool checkParticleLife() {
    const int n = 1024, steps = 10;
    const float dt = 0.02f, rmax = 0.1f, beta = 0.3f, friction = 0.9f, forceScale = 10.0f;
    std::vector<Body> bodies(n), a(n), b(n);
    uint32_t seed = 2024u;
    for (int i = 0; i < n; ++i) {
        bodies[i].pos = make_float2(gv::random01(seed), gv::random01(seed));
        bodies[i].vel = make_float2(0.0f, 0.0f);
        bodies[i].type = i % 4;
        bodies[i].id = i;
        bodies[i].pad = make_float2(0.0f, 0.0f);
    }
    a = bodies;
    for (int s = 0; s < steps; ++s) {
        hostPlStep(a, b, dt, rmax, beta, friction, forceScale);
        a.swap(b);
    }

    Body* dIn = gv::upload(bodies);
    Body* dOut = gv::upload(bodies);
    for (int s = 0; s < steps; ++s) {
        pl_step<<<(n + 127) / 128, 128>>>(dIn, dOut, n, dt, rmax, beta, friction, forceScale);
        Body* t = dIn;
        dIn = dOut;
        dOut = t;
    }
    std::vector<Body> got(n);
    gv::download(got, dIn);
    cudaFree(dIn);
    cudaFree(dOut);

    int good = 0;
    for (int i = 0; i < n; ++i) {
        const bool identity = got[i].id == i && got[i].type == i % 4; // the struct travelled intact
        const bool near = torusDistance(got[i].pos.x, a[i].pos.x, 1.0f) < 2e-3f && torusDistance(got[i].pos.y, a[i].pos.y, 1.0f) < 2e-3f &&
                          std::fabs(got[i].vel.x - a[i].vel.x) < 2e-2f && std::fabs(got[i].vel.y - a[i].vel.y) < 2e-2f;
        if (!identity) {
            std::printf("    particle %d lost its identity: id %d type %d\n", i, got[i].id, got[i].type);
            return false;
        }
        if (near) ++good;
    }
    if (good < n * 99 / 100) {
        std::printf("    only %d of %d particles match the host simulation\n", good, n);
        return false;
    }
    return true;
}

bool checkPrimordial() {
    const int n = 600, steps = 5;
    const float size = 40.0f, radius = 5.0f, alpha = 3.14159265f, beta = 0.2967f, speed = 0.67f;
    std::vector<Pps> init(n), a(n), b(n);
    uint32_t seed = 31337u;
    for (int i = 0; i < n; ++i) {
        init[i].pos = make_float2(gv::random01(seed) * size, gv::random01(seed) * size);
        init[i].phi = gv::random01(seed) * 6.2831853f;
        init[i].neighbors = 0;
    }
    a = init;
    for (int s = 0; s < steps; ++s) {
        hostPpsStep(a, b, size, radius, alpha, beta, speed);
        a.swap(b);
    }

    Pps* dIn = gv::upload(init);
    Pps* dOut = gv::upload(init);
    for (int s = 0; s < steps; ++s) {
        pps_step<<<(n + 127) / 128, 128>>>(dIn, dOut, n, size, radius, alpha, beta, speed);
        Pps* t = dIn;
        dIn = dOut;
        dOut = t;
    }
    std::vector<Pps> got(n);
    gv::download(got, dIn);
    cudaFree(dIn);
    cudaFree(dOut);

    int good = 0;
    long hostNeighbors = 0, gotNeighbors = 0;
    for (int i = 0; i < n; ++i) {
        hostNeighbors += a[i].neighbors;
        gotNeighbors += got[i].neighbors;
        if (torusDistance(got[i].pos.x, a[i].pos.x, size) < 2e-2f && torusDistance(got[i].pos.y, a[i].pos.y, size) < 2e-2f &&
            std::fabs(got[i].phi - a[i].phi) < 5e-2f) {
            ++good;
        }
    }
    if (good < n * 97 / 100) {
        std::printf("    only %d of %d particles match the host simulation\n", good, n);
        return false;
    }
    if (std::labs(hostNeighbors - gotNeighbors) > hostNeighbors / 100) {
        std::printf("    neighbour counts differ: host %ld, device %ld\n", hostNeighbors, gotNeighbors);
        return false;
    }
    return true;
}

} // namespace

int runParticleChecks(bool useGpu) {
    const gv::Check checks[] = {{"particle life: 1024 bodies, 10 steps", checkParticleLife},
                                {"primordial particle system: 600, 5 steps", checkPrimordial}};
    return gv::runChecks("particle systems", checks, 2, useGpu);
}

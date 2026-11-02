// Struct in buffer (read-write POD struct buffer parameter)
struct Particle {
    float4 pos;
    float4 vel;
};

__global__ void update_particles(Particle* particles, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        particles[i].pos.x += particles[i].vel.x * dt;
        particles[i].pos.y += particles[i].vel.y * dt;
        particles[i].pos.z += particles[i].vel.z * dt;
    }
}

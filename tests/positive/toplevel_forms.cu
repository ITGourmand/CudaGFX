// Top-level forms: namespace, extern "C", qualified launch, qualifier macro,
// __launch_bounds__ with two arguments, non-POD host class (ignored), local constexpr.
#include <cuda_runtime.h>
#include <vector>
#define KERNEL __global__
#define DEVFN __host__ __device__

class HostOnly {            // non-POD: methods, never seen on the device side
public:
    HostOnly() : v_(3) {}
    int get() const { return v_; }
private:
    int v_;
};

struct Pod { float a; float b; };   // POD: usable on the device side

namespace gpu {
DEVFN inline float half_of(float v) { return v * 0.5f; }

KERNEL __launch_bounds__(128, 2) void in_namespace(float* y, float a, float b) {
    constexpr float k = 4.f;
    Pod p;
    p.a = a;
    p.b = b;
    y[threadIdx.x] = half_of(p.a) + p.b * k;
}
} // namespace gpu

extern "C" {
KERNEL void in_extern_c(int* y) { y[threadIdx.x] = (int)threadIdx.x; }
}

int main() {
    HostOnly h;
    std::vector<float> v(128);
    float* d = nullptr;
    int* di = nullptr;
    cudaMalloc((void**)&d, 128 * sizeof(float));
    cudaMalloc((void**)&di, 128 * sizeof(int));
    gpu::in_namespace<<<1, 128>>>(d, 1.f, 2.f);
    in_extern_c<<<2, 64>>>(di);
    cudaFree(d);
    cudaFree(di);
    return h.get() - 3;
}

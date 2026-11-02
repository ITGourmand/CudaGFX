// gv_common.h: helpers shared by the .cu files of the GPU validation example. It must be included AFTER
// <cuda_runtime.h> (it uses the cudagfx runtime) and holds host code only, never a kernel.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace gv {

// Relative / absolute comparison of two floats.
inline bool close(float a, float b, float tolerance = 1e-4f) { return std::fabs(a - b) <= tolerance * (1.0f + std::fabs(b)); }

// Deterministic pseudo-random numbers (the same sequence on every machine).
inline uint32_t nextRandom(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}
inline float random01(uint32_t& state) { return static_cast<float>(nextRandom(state) >> 8) * (1.0f / 16777216.0f); }

template <typename T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(T));
    cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return device;
}

template <typename T>
void download(std::vector<T>& host, const T* device) {
    cudaMemcpy(host.data(), device, host.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

struct Check {
    const char* name;
    bool (*run)();
};

// Runs every check on the GPU (if `useGpu`) and on the CPU backend. A check passes when it returns true AND the
// runtime recorded no error. Prints the time of each run: a GPU that is slower than the CPU backend on the big
// kernels is a finding too.
inline int runChecks(const char* suite, const Check* checks, int count, bool useGpu) {
    std::printf("-- %s\n", suite);
    int failures = 0;
    for (int i = 0; i < count; ++i) {
        for (int pass = 0; pass < 2; ++pass) {
            const bool gpu = pass == 0;
            if (gpu && !useGpu) continue;
            cudagfx::setExecutionMode(gpu ? cudagfx::ExecutionMode::GPU : cudagfx::ExecutionMode::CPU);
            cudaGetLastError();
            const auto start = std::chrono::steady_clock::now();
            const bool ok = checks[i].run();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            const std::string message = cudagfx::lastErrorMessage(); // before cudaGetLastError() clears it
            const cudaError_t err = cudaGetLastError();
            const bool passed = ok && err == cudaSuccess;
            std::printf("%s  %-46s [%s] %8.1f ms\n", passed ? "PASS" : "FAIL", checks[i].name, gpu ? "GPU" : "CPU", ms);
            if (err != cudaSuccess) std::printf("    runtime error %s: %s\n", cudaGetErrorString(err), message.c_str());
            if (!passed) ++failures;
        }
    }
    return failures;
}

} // namespace gv

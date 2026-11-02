// gpu_checks.h: the suites of the GPU validation example (declarations only, so that main.cpp does not depend on the
// generated code). Each suite runs its checks on the GPU (when `useGpu`) and on the CPU backend, prints one PASS / FAIL
// line per check and returns the number of failures.
#pragma once

int runGpuChecks(bool useGpu);       // gpu_checks.cu:      the basics (arguments, struct buffer, barriers, atomics, swizzles)
int runLifeChecks(bool useGpu);      // life_checks.cu:     Conway's Game of Life (global and tiled shared-memory kernels)
int runParticleChecks(bool useGpu);  // particle_checks.cu: Particle Life and the Primordial Particle System
int runAlgorithmChecks(bool useGpu); // algorithm_checks.cu: Gray-Scott, bitonic sort, scan, tiled matmul, Mandelbrot

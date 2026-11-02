// CpuEmitter.hpp: C++ implementation of the kernels for the CPU backend (--emit-cpu).
//
// The CPU backend is a second translation of the kernel bodies (the first one targets shaders): it prints the
// typed AST as C++17, to be compiled together with the host code. It is a different dialect from the shaders
// (the CUDA vector types and built-ins are mapped by cudagfx_cpu.h instead of being renamed), which is why it
// has its own emitter rather than options in ShaderEmitter.
#pragma once

#include <string>
#include "AST.hpp"

namespace cudagfx {

// Declarations of the CPU entry point of every kernel (`cudagfx_gen::cpu::<kernel>(grid, blk, args...)`), for
// the kernels header. The definitions live in the CPU header, which the host code includes at its very end:
// the structs, constants and __constant__ arrays the kernels use are only complete there.
std::string emitCpuDeclarations(const ParsedModule& module);

// The CPU header: kernel bodies, their __device__ helpers and the entry points.
// `cpuRuntimeInclude` is the include path of cudagfx_cpu.h.
std::string emitCpuHeader(const ParsedModule& module, const std::string& sourceName, const std::string& cpuRuntimeInclude);

} // namespace cudagfx

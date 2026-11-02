// KernelAbi.hpp: binding of a kernel's parameters to the ABI (buffer slots, uniform lanes).
//
// SINGLE source of the binding: the shader emitter (which declares where to read) and the host emitter
// (which generates the packing code) both compute it here, from abi::ArgLayout, the algorithm shared with the runtime.
#pragma once

#include <string>
#include <vector>
#include "AST.hpp"

namespace cudagfx {

struct ParamBinding {
    const Parameter* param = nullptr;
    bool isBuffer = false;
    unsigned bufferSlot = 0;   // buffers: compute binding slot
    unsigned components = 0;   // by-value parameters: 1 to 4 32-bit words
    unsigned lane = 0;         // by-value parameters: absolute starting lane in u_<k>_args
};

struct KernelBindings {
    std::vector<ParamBinding> params;
    unsigned argSlots = 0;     // size of the u_<k>_args array (in vec4)
    unsigned bufferCount = 0;
    bool ok = true;
    std::string error;
};

inline constexpr unsigned kMaxBufferBindings = 16;

KernelBindings computeBindings(const FunctionDecl& kernel);

// Size in bytes of a numeric scalar or vector as the host compiler lays it out (float3 is 12 bytes), 0 for the types
// that cannot travel through a constant uniform array (bool, structs...).
unsigned constantElementBytes(const Type& element);

// The __constant__ objects a kernel (and the __device__ functions it calls) read from uniform arrays, kernel first.
std::vector<const GlobalConstDecl*> uniformConstants(const ParsedModule& module, const FunctionDecl& kernel);

} // namespace cudagfx

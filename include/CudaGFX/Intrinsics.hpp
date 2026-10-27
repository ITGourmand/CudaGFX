// Intrinsics.hpp: native CUDA functions translated to the shader language (single table).
#pragma once

#include <string>
#include "AST.hpp"

namespace cudagfx {

struct Intrinsic {
    enum class Shape {
        FloatUnary,   // float f(float)
        FloatBinary,  // float f(float, float)
        Generic,      // min / max / abs: common type of the arguments
        VectorMake,   // make_float4(x, y, z, w)
        Fixed         // arguments and result of fixed types (bit casts, popcount, isnan...)
    };

    std::string cudaName;
    std::string shaderName;
    Shape shape;
    unsigned argc;
    Type::Kind argKind;   // Fixed: type of each argument
    Type::Kind retKind;   // Fixed: result type; VectorMake: scalar type of the components
    unsigned components;  // VectorMake: number of components
    bool warnSemantics;   // fmodf: the sign of the remainder differs from the native mod (CGFX0020)
};

// Name under which __popc is emitted. bitCount exists in GLSL only (HLSL, hence SPIR-V, Metal, WGSL and DXIL, calls it
// countbits) and bgfx_shader.sh defines neither: the emitter declares this macro in the shader header when needed.
inline constexpr const char* kPopcountShaderName = "cgfx_popcount";

// nullptr if `name` is not a supported native function.
const Intrinsic* findIntrinsic(const std::string& name);

// Families rejected with a dedicated diagnostic.
bool isWarpIntrinsic(const std::string& name);          // CGFX0003
bool isUnsupportedDeviceFunction(const std::string& name); // CGFX0005 (printf, malloc, cuda*, ...)

// Supported CUDA atomics. Returns false if `name` is not one of them.
bool parseAtomicName(const std::string& name, AtomicOp& op);

} // namespace cudagfx

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

// Natives that GLSL and HLSL spell differently (bgfx compiles the .sc as HLSL for SPIR-V, Metal, WGSL and DXIL) or that
// need an expression: emitted as macros in the header of a shader that calls them. bgfx_shader.sh defines none of them.
inline constexpr const char* kPopcountShaderName = "cgfx_popcount";
inline constexpr const char* kFmaShaderName = "cgfx_fma";
inline constexpr const char* kFdivShaderName = "cgfx_fdiv";
inline constexpr const char* kCopysignShaderName = "cgfx_copysign";
inline constexpr const char* kIsfiniteShaderName = "cgfx_isfinite";

struct ShaderHelperMacro {
    const char* name;
    const char* glsl; // parameters and body, e.g. "(x) bitCount(x)"
    const char* hlsl;
};

inline constexpr ShaderHelperMacro kShaderHelperMacros[] = {
    {kPopcountShaderName, "(x) bitCount(x)", "(x) countbits(x)"},
    {kFmaShaderName, "(a, b, c) fma(a, b, c)", "(a, b, c) mad(a, b, c)"},
    {kFdivShaderName, "(a, b) ((a) / (b))", "(a, b) ((a) / (b))"},
    {kCopysignShaderName, "(a, b) uintBitsToFloat((floatBitsToUint(a) & 0x7FFFFFFFu) | (floatBitsToUint(b) & 0x80000000u))",
     "(a, b) uintBitsToFloat((floatBitsToUint(a) & 0x7FFFFFFFu) | (floatBitsToUint(b) & 0x80000000u))"},
    // Exponent bits, not isinf/isnan: Tint (WGSL) does not implement OpIsInf / OpIsNan.
    {kIsfiniteShaderName, "(x) ((floatBitsToUint(x) & 0x7F800000u) != 0x7F800000u)",
     "(x) ((floatBitsToUint(x) & 0x7F800000u) != 0x7F800000u)"},
};

// True if `shaderName` is the name of one of kShaderHelperMacros.
bool isShaderHelperMacro(const std::string& shaderName);

// nullptr if `name` is not a supported native function.
const Intrinsic* findIntrinsic(const std::string& name);

// Families rejected with a dedicated diagnostic.
bool isWarpIntrinsic(const std::string& name);          // CGFX0003
bool isUnsupportedDeviceFunction(const std::string& name); // CGFX0005 (printf, malloc, cuda*, ...)

// Supported CUDA atomics. Returns false if `name` is not one of them.
bool parseAtomicName(const std::string& name, AtomicOp& op);

} // namespace cudagfx

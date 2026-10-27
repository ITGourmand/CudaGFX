// ShaderNames.cpp
#include "ShaderNames.hpp"

#include <unordered_set>

namespace cudagfx {

namespace {

const std::unordered_set<std::string>& reservedSet() {
    static const std::unordered_set<std::string> set = {
        // --- GLSL: keywords
        "attribute", "const", "uniform", "varying", "buffer", "shared", "coherent", "volatile", "restrict",
        "readonly", "writeonly", "atomic_uint", "layout", "centroid", "flat", "smooth", "noperspective",
        "patch", "sample", "break", "continue", "do", "for", "while", "switch", "case", "default", "if",
        "else", "subroutine", "in", "out", "inout", "int", "void", "bool", "true", "false", "float",
        "double", "discard", "return", "uint", "struct", "lowp", "mediump", "highp", "precision",
        "invariant", "precise", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "uvec2", "uvec3",
        "uvec4", "bvec2", "bvec3", "bvec4", "dvec2", "dvec3", "dvec4", "mat2", "mat3", "mat4", "mat2x2",
        "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4", "mat4x2", "mat4x3", "mat4x4", "dmat2", "dmat3",
        "dmat4", "sampler1D", "sampler2D", "sampler3D", "samplerCube", "sampler2DArray", "image1D",
        "image2D", "image3D", "imageCube", "uimage2D", "iimage2D",
        // --- GLSL: reserved words
        "common", "partition", "active", "asm", "class", "union", "enum", "typedef", "template", "this",
        "resource", "goto", "inline", "noinline", "public", "static", "extern", "external", "interface",
        "long", "short", "half", "fixed", "unsigned", "superp", "input", "output", "hvec2", "hvec3",
        "hvec4", "fvec2", "fvec3", "fvec4", "filter", "sizeof", "cast", "namespace", "using", "main",
        // --- HLSL / MSL / WGSL
        "texture", "Texture2D", "sampler", "SamplerState", "register", "cbuffer", "tbuffer", "string",
        "vector", "matrix", "float2", "float3", "float4", "int2", "int3", "int4", "uint2", "uint3",
        "uint4", "min16float", "min10float", "min16int", "min16uint", "linear", "row_major",
        "column_major", "compile", "decl", "pass", "technique", "Buffer", "RWBuffer", "StructuredBuffer",
        "RWStructuredBuffer", "ByteAddressBuffer", "RWByteAddressBuffer", "groupshared", "nointerpolation",
        "snorm", "unorm", "triangle", "point", "line", "typename", "kernel", "thread", "device",
        "constant", "threadgroup", "vertex", "fragment", "metal", "ushort", "uchar", "fn", "let", "var",
        "array", "type", "loop", "override", "alias", "ref", "ptr", "i32", "u32", "f32", "f16",
        "private", "workgroup", "storage", "function", "bitcast", "enable", "diagnostic",
        // --- Built-in functions (a local variable of the same name would shadow their use by the generated code)
        "abs", "min", "max", "clamp", "mix", "step", "smoothstep", "mod", "pow", "exp", "exp2", "log",
        "log2", "sqrt", "inversesqrt", "floor", "ceil", "trunc", "round", "roundEven", "fract", "sign",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "dot", "cross",
        "length", "normalize", "distance", "reflect", "refract", "faceforward", "transpose",
        "determinant", "inverse", "mul", "saturate", "lerp", "frac", "rsqrt", "ddx", "ddy", "fwidth",
        "any", "all", "not", "equal", "notEqual", "lessThan", "lessThanEqual", "greaterThan",
        "greaterThanEqual", "isnan", "isinf", "fma", "mad", "radians", "degrees", "barrier",
        "memoryBarrier", "memoryBarrierShared", "memoryBarrierBuffer", "groupMemoryBarrier",
        "floatBitsToInt", "floatBitsToUint", "intBitsToFloat", "uintBitsToFloat", "bitCount",
        "findLSB", "findMSB", "bitfieldExtract", "bitfieldInsert", "bitfieldReverse", "texelFetch",
        "imageLoad", "imageStore", "atomicAdd", "atomicMin", "atomicMax", "atomicAnd", "atomicOr",
        "atomicXor", "atomicExchange", "atomicCompSwap", "atomicFetchAndAdd", "atomicFetchAndAnd",
        "atomicFetchAndMax", "atomicFetchAndMin", "atomicFetchAndOr", "atomicFetchAndXor",
        "atomicFetchAndExchange", "atomicFetchCompareExchange", "InterlockedAdd", "InterlockedMin",
        "InterlockedMax", "InterlockedAnd", "InterlockedOr", "InterlockedXor", "InterlockedExchange",
        "InterlockedCompareExchange", "pack", "unpack", "select",
        // --- bgfx macros used by the generated code
        "SHARED", "BUFFER_RO", "BUFFER_RW", "BUFFER_WO", "NUM_THREADS", "CONST", "BRANCH", "LOOP",
        "UNROLL", "FORMAT", "WRITEONLY", "mat3", "vec4_splat", "mtxFromRows", "mtxFromCols",
    };
    return set;
}

bool startsWith(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

} // namespace

bool isReservedShaderIdentifier(const std::string& name) {
    return reservedSet().count(name) != 0;
}

std::string safeIdentifier(const std::string& name) {
    return isReservedShaderIdentifier(name) ? "cgfx_" + name : name;
}

bool isForbiddenIdentifier(const std::string& name) {
    return startsWith(name, "gl_") || startsWith(name, "bgfx_") || startsWith(name, "cgfx_") ||
           startsWith(name, "u_cgfx") || name.find("__") != std::string::npos;
}

} // namespace cudagfx

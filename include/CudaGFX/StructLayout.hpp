// StructLayout.hpp: memory layout of the POD structs that travel through a buffer.
//
// A struct used as a buffer element is described three times: by the host C++ compiler (what `sizeof` and
// `offsetof` see), by GLSL std430 (OpenGL, Vulkan, WebGPU through shaderc) and by HLSL structured buffers
// (Direct3D), whose packing is tight. The element only works if the three agree: otherwise the shader reads
// the right bytes at the wrong offsets, silently. This module computes the three layouts and explains the
// first disagreement, so that the transpiler can refuse the struct (CGFX0021) instead of emitting a wrong shader.
#pragma once

#include <functional>
#include <string>
#include <vector>
#include "AST.hpp"

namespace cudagfx {

enum class LayoutRules {
    Host,   // C++ with the CUDA vector alignments (float2: 8, float4: 16)
    Std430, // GLSL std430
    Tight   // HLSL structured buffer: every member aligned on 4 bytes only
};

struct StructLayout {
    unsigned size = 0;
    unsigned align = 4;
    std::vector<unsigned> offsets; // one per field
    bool valid = true;             // false: a field has no layout (bool, 3-component vector, array...)
    std::string problem;           // why `valid` is false
};

using StructLookup = std::function<const StructDecl*(const std::string&)>;

StructLayout computeLayout(const StructDecl& decl, LayoutRules rules, const StructLookup& lookup);

// Empty string if `decl` (and every struct nested in it) has the same layout under the three rules and fits in
// `maxBytes`; otherwise a one-line explanation naming the struct and the first offending field.
std::string portableLayoutProblem(const StructDecl& decl, const StructLookup& lookup, unsigned maxBytes);

// `static_assert`s (sizeof / alignof / offsetof) appended to the generated host code: they make the C++ compiler
// confirm, with the real definition of each struct, the layout the shader was generated for.
std::string emitHostLayoutAsserts(const std::vector<const StructDecl*>& structs, const StructLookup& lookup);

} // namespace cudagfx

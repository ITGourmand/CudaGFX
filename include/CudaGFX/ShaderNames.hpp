// ShaderNames.hpp: protecting user identifiers against the target languages.
//
// A CUDA program may legitimately name a variable `input`, `output`, `sample`, `filter`,
// `half`, `mod`... These names are reserved words (or built-in functions) in GLSL, HLSL, MSL or
// WGSL: the generated shader would be invalid. The emitter therefore renames these identifiers.
#pragma once

#include <string>

namespace cudagfx {

// True if `name` is a reserved word or a built-in function of one of the target languages, or a bgfx
// macro used by the generated code.
bool isReservedShaderIdentifier(const std::string& name);

// Emission name of a user identifier: unchanged, or prefixed with "cgfx_" if it is reserved.
std::string safeIdentifier(const std::string& name);

// Identifiers the user may not use at all (CGFX0016): gl_, bgfx_, cgfx_, u_cgfx prefixes
// or a double underscore (reserved in GLSL).
bool isForbiddenIdentifier(const std::string& name);

} // namespace cudagfx

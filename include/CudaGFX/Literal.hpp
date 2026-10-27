// Literal.hpp: C++ numeric literal parsing and canonical formatting for shaders.
#pragma once

#include <cstdint>
#include <string>

namespace cudagfx {

struct NumericLiteral {
    enum class Kind {
        Int,     // 32-bit int
        UInt,    // 32-bit unsigned int
        Int64,   // long / long long / value > 32 bits (not supported by CudaGFX)
        Float,   // 32-bit float (an unsuffixed literal is treated as float too, see README)
        Invalid
    };
    Kind kind = Kind::Invalid;
    std::uint64_t intValue = 0;
    double floatValue = 0.0;
    bool hasFloatSuffix = false; // 1.0f
};

// `text` is the token text (for example "0xFFu", "1.f", "2.", "1e-3f", "1'000").
NumericLiteral parseNumericLiteral(const std::string& text);

// Value of a character literal ('a', '\n', '\x41', '\0'). Returns false if unsupported.
bool parseCharLiteral(const std::string& text, std::int64_t& value);

// Form accepted by every GLSL/HLSL/MSL/WGSL backend: never an `f` suffix, always a
// decimal point or an exponent ("1.0", "0.001"). The value is rounded to 32 bits.
std::string formatFloatLiteral(double value);

} // namespace cudagfx

// cudagfx_abi.h: ABI contract shared between the `cudagfx` tool and the runtime (specification constraint C6).
//
// The layout of kernel parameters is defined ONLY here. The tool uses it to generate the shader, the
// runtime uses it to fill the uniform. C++17, standard library only, no exceptions, no RTTI.

//
// Packing rules (section 8.2 of the specification, CudaGFX_spec_prompt.md):
//   - every scalar occupies one 32-bit lane of a vec4;
//   - int / uint / bool are passed by bit-cast (binary copy, no conversion);
//   - a 2-component vector starts on an even lane;
//   - a 3- or 4-component vector starts on lane 0 of a slot;
//   - no parameter straddles two vec4.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace cudagfx {
namespace abi {

static_assert(sizeof(float) == 4, "CudaGFX assumes 32-bit floats");

// Names of the uniforms generated for a kernel `k`:
//   uniform vec4 u_<k>_args[N];   packed parameters
//   uniform vec4 u_<k>_grid;      gridDim (xyz as bit-cast uint), declared only if the kernel uses it
// (the specification does not say where gridDim lives: see docs/verifications.md)
inline constexpr const char* kArgsSuffix = "_args";
inline constexpr const char* kGridSuffix = "_grid";

inline constexpr unsigned kLanesPerSlot = 4;
inline constexpr unsigned kMaxArgSlots = 16;
inline constexpr unsigned kMaxArgLanes = kMaxArgSlots * kLanesPerSlot;

// Starting alignment, in lanes, according to the number of 32-bit components of the parameter.
constexpr unsigned alignLanesFor(unsigned components) {
    return components <= 1 ? 1u : (components == 2 ? 2u : 4u);
}

// Placement of the parameters in the lanes: the SINGLE definition of the algorithm, shared by the shader
// emitter (which declares where to read each parameter) and by the runtime packer (which writes the values).
class ArgLayout {
public:
    // Places a parameter of `components` lanes (1 to 4). Returns false on overflow or on an invalid
    // number of components; otherwise `startLane` receives the absolute starting lane.
    bool place(unsigned components, unsigned& startLane) {
        if (components == 0 || components > 4) {
            failed_ = true;
            return false;
        }
        const unsigned a = alignLanesFor(components);
        unsigned start = cursor_;
        const unsigned rem = start % a;
        if (rem != 0) start += a - rem;
        if (start + components > kMaxArgLanes) {
            failed_ = true;
            return false;
        }
        startLane = start;
        cursor_ = start + components;
        return true;
    }

    unsigned slotCount() const { return (cursor_ + kLanesPerSlot - 1) / kLanesPerSlot; }
    unsigned cursorLane() const { return cursor_; }
    bool failed() const { return failed_; }

private:
    unsigned cursor_ = 0;
    bool failed_ = false;
};

// GLSL/HLSL component selector to read a parameter: ".x"..".w", ".xy", ".zw", ".xyz",
// or "" (whole slot) for 4 components. nullptr if (lane, components) violates the alignment rules.
inline const char* laneSwizzle(unsigned laneInSlot, unsigned components) {
    switch (components) {
        case 1: {
            static const char* const one[4] = {".x", ".y", ".z", ".w"};
            return laneInSlot < 4 ? one[laneInSlot] : nullptr;
        }
        case 2:
            return laneInSlot == 0 ? ".xy" : (laneInSlot == 2 ? ".zw" : nullptr);
        case 3:
            return laneInSlot == 0 ? ".xyz" : nullptr;
        case 4:
            return laneInSlot == 0 ? "" : nullptr;
        default:
            return nullptr;
    }
}

// Names of the uniforms of a kernel (used both by the generated shader and by the runtime).
inline std::string argsUniformName(const std::string& kernel) { return "u_" + kernel + kArgsSuffix; }
inline std::string gridUniformName(const std::string& kernel) { return "u_" + kernel + kGridSuffix; }

// Host-side packer. The lanes are stored in an array of float, but only the binary
// representation matters: an integer is never converted to float.
class ArgPacker {
public:
    ArgPacker() { std::memset(lanes_, 0, sizeof(lanes_)); }

    bool pushFloat(float v) { return pushWords(&v, 1); }
    bool pushInt(std::int32_t v) { return pushWords(&v, 1); }
    bool pushUInt(std::uint32_t v) { return pushWords(&v, 1); }
    bool pushBool(bool v) {
        const std::uint32_t w = v ? 1u : 0u;
        return pushWords(&w, 1);
    }

    // `data` points to `components` 32-bit words (1 to 4). Returns false and sets
    // failed() on overflow or on an invalid number of components.
    bool pushWords(const void* data, unsigned components) {
        unsigned start = 0;
        if (!layout_.place(components, start)) return false;
        std::memcpy(&lanes_[start], data, components * sizeof(float));
        return true;
    }

    unsigned slotCount() const { return layout_.slotCount(); }
    unsigned cursorLane() const { return layout_.cursorLane(); }
    const float* data() const { return lanes_; }
    bool failed() const { return layout_.failed(); }

private:
    float lanes_[kMaxArgLanes];
    ArgLayout layout_;
};

}  // namespace abi
}  // namespace cudagfx

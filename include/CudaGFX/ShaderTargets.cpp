// ShaderTargets.cpp
#include "ShaderTargets.hpp"

#include <cctype>

namespace cudagfx {

namespace {

bool startsWith(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool allDigits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

// Returns the backend of a shaderc profile, or "" if the runtime cannot use it.
std::string backendForProfile(const std::string& profile) {
    if (startsWith(profile, "spirv")) return "spv";
    if (startsWith(profile, "metal")) return "mtl";
    if (profile == "wgsl") return "wgsl";
    if (startsWith(profile, "s_4_") || startsWith(profile, "s_5_")) return "dxbc";
    if (startsWith(profile, "s_6_")) return "dxil";
    if (endsWith(profile, "_es") && allDigits(profile.substr(0, profile.size() - 3))) return "essl";
    if (allDigits(profile)) return "glsl";
    return {};
}

} // namespace

bool parseShaderTarget(const std::string& spec, ShaderTarget& out, std::string& error) {
    const auto colon = spec.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= spec.size()) {
        error = "target '" + spec + "' is invalid: expected format platform:profile (for example linux:spirv)";
        return false;
    }
    out.platform = spec.substr(0, colon);
    out.profile = spec.substr(colon + 1);
    out.backend = backendForProfile(out.profile);
    if (out.backend.empty()) {
        error = "shaderc profile '" + out.profile +
                "' is not supported by the runtime (profiles: spirv*, 120-450, *_es, s_4_x/s_5_x, s_6_x, metal*, wgsl)";
        return false;
    }
    return true;
}

std::vector<ShaderTarget> defaultShaderTargets() {
    return {{"linux", "spirv", "spv"}, {"linux", "430", "glsl"}};
}

std::string shaderSymbolName(const std::string& kernel, const BlockVariant& variant, const std::string& backend) {
    return "cs_" + kernel + "_" + variant.suffix() + "_" + backend;
}

} // namespace cudagfx

// ShaderTargets.hpp: mapping between shaderc profiles and bgfx backends (embedded_shader.h convention).
//
//   backend | bgfx renderer           | symbol suffix      | shaderc profiles
//   --------+--------------------------+--------------------+------------------------
//   spv     | Vulkan                   | _spv               | spirv, spirv10-10...
//   glsl    | OpenGL                   | _glsl              | 120..450
//   essl    | OpenGLES                 | _essl              | 300_es, 310_es...
//   dxbc    | Direct3D11               | _dxbc              | s_4_0, s_5_0
//   dxil    | Direct3D12               | _dxil              | s_6_0 .. s_6_x
//   mtl     | Metal                    | _mtl               | metal, metal10-10...
//   wgsl    | WebGPU                   | _wgsl              | wgsl
#pragma once

#include <string>
#include <vector>
#include "Variants.hpp"

namespace cudagfx {

struct ShaderTarget {
    std::string platform; // value of shaderc's --platform (linux, windows, osx, ...)
    std::string profile;  // value of --profile
    std::string backend;  // spv, glsl, essl, dxbc, dxil, mtl, wgsl
};

// Parses "platform:profile" (for example "linux:spirv", "windows:s_5_0").
bool parseShaderTarget(const std::string& spec, ShaderTarget& out, std::string& error);

// Targets used when --shaderc is given without --target: Vulkan (SPIR-V) and OpenGL 4.3.
std::vector<ShaderTarget> defaultShaderTargets();

// Name of the C array (and of the .h file without extension): cs_<kernel>_<variant>_<backend>.
std::string shaderSymbolName(const std::string& kernel, const BlockVariant& variant, const std::string& backend);

// Matching field of cudagfx::ShaderBinaryTable (same as the backend suffix).
inline const char* shaderTableField(const std::string& backend) { return backend.c_str(); }

} // namespace cudagfx

// HostEmitter.hpp: rewriting of the host code (.cu -> standard C++17 .cpp) and generation of the stubs.
#pragma once

#include <string>
#include <vector>
#include "AST.hpp"
#include "Diagnostics.hpp"
#include "Preprocessor.hpp"
#include "Variants.hpp"

namespace cudagfx {

struct HostRewriteConfig {
    std::string kernelsHeaderInclude; // text of the #include "..." that replaces #include <cuda_runtime.h>
    std::string sourceName;           // name of the .cu, for the #line directive if the include has to be added
    bool preserveLines = true;        // C7: unmodified lines keep their line number
};

// Rewrites the main file: removes kernels and __device__ functions, replaces every launch with a call
// to the stub, substitutes the CUDA include. All the rest of the text is copied verbatim.
// Returns false (and reports CGFX0100) if two modifications overlap.
bool rewriteHostSource(const std::string& original, const ParsedModule& module, const Preprocessor& pp,
                       const HostRewriteConfig& config, DiagnosticsEngine& diags, std::string& out);

// Compiled shader file, to be referenced from the stubs header.
struct ShaderArtifact {
    std::string kernel;
    BlockVariant variant;
    std::string backend;      // spv, glsl, ...
    std::string symbol;       // name of the C array
    std::string includePath;  // path of the .h header, relative to the stubs header
};

struct KernelsHeaderConfig {
    std::string sourceName;
    std::string runtimeInclude = "CudaGFX/cudagfx_rt.h";
};

std::string emitKernelsHeader(const ParsedModule& module, const VariantMap& variants,
                              const std::vector<ShaderArtifact>& artifacts, const KernelsHeaderConfig& config);

} // namespace cudagfx

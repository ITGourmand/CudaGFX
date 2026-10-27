// Compiler.hpp: orchestration of the tool (.cu file -> shaders + host code + stubs).
#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "ShaderTargets.hpp"

namespace cudagfx {

// Exit codes (contract with the build systems).
enum ExitCode : int {
    kExitOk = 0,
    kExitSourceError = 1,    // error in the CUDA source (CGFXnnnn diagnostics)
    kExitUsageError = 2,     // incorrect usage, unreadable input file
    kExitShadercFailure = 3, // shaderc failed
    kExitInternalError = 4   // internal error or unable to write
};

struct CompilerOptions {
    std::string inputFilePath;
    std::string outputHostPath;
    std::string outputKernelsHeaderPath;
    std::string outputShaderPath;     // direct path, or a pattern containing '%' (replaced by the kernel name)
    std::string outputShaderDir;
    std::string shadercPath;
    std::vector<std::string> shadercIncludes; // -i: shaderc include directories
    std::vector<std::string> includePaths;    // -I: preprocessor include directories (#include "x.h")
    std::vector<ShaderTarget> targets;
    bool embed = false;
    bool dumpIr = false;
    bool noLinePreserve = false;
    std::string runtimeInclude = "CudaGFX/cudagfx_rt.h";
    std::map<std::string, std::string> defines;
    std::string dependsPath;          // --depends: dependency file in Makefile format
};

class Compiler {
public:
    // Every exception is caught: the caller only ever receives an exit code.
    int run(const CompilerOptions& options, std::ostream& out = std::cout, std::ostream& err = std::cerr) const;
};

} // namespace cudagfx

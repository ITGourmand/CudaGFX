// cudagfx.cpp: command-line front-end of the cudagfx tool.
//
// This file only parses the arguments into a cudagfx::CompilerOptions and hands it to cudagfx::Compiler;
// all the real work (preprocessing, parsing, semantic analysis, code generation, shaderc) lives in
// include/CudaGFX/. Exit codes are defined by cudagfx::ExitCode (see Compiler.hpp).
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "CudaGFX/Compiler.hpp"

namespace {

void printUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " <input.cu> [options]\n\n"
        << "Outputs:\n"
        << "  -o <file>                    Generated C++17 host code (kernels and launches are rewritten)\n"
        << "  --kernels-header <file>      Stubs header (default: <-o without extension>_kernels.h)\n"
        << "  --shader-out <path>          .sc shader; a pattern containing '%' is replaced by the kernel name\n"
        << "  --shader-dir <dir>           Shader directory (default: .), names cs_<kernel>_<block>.sc\n\n"
        << "shaderc:\n"
        << "  --shaderc <path>             shaderc executable: compiles every shader for every target\n"
        << "  --target <platform:profile>  shaderc target, repeatable (default: linux:spirv and linux:430)\n"
        << "  --embed                      Embed the binaries in the header (C arrays, --bin2c)\n"
        << "  -i <dir>                     Include directory passed to shaderc (bgfx_compute.sh...)\n\n"
        << "Preprocessor:\n"
        << "  -I <dir>                     Search directory for #include \"file.h\"\n"
        << "  -D<name>[=<value>]           Defines a macro (value 1 by default)\n\n"
        << "Miscellaneous:\n"
        << "  --depends <file>             Writes the dependencies (followed headers) in Makefile format\n"
        << "  --runtime-include <path>     Text of the runtime #include in the generated header\n"
        << "  --no-line-preserve           Do not preserve the line numbering of the host code\n"
        << "  --dump-ir                    Writes a JSON summary (kernels, blocks, buffers) to standard output\n"
        << "  -h, --help                   This help\n\n"
        << "Exit codes: 0 success, 1 error in the source, 2 usage, 3 shaderc failure, 4 internal error\n";
}

// Returns the value of option `arg`, either attached (-Ifoo) or in the next argument.
bool takeValue(int argc, char** argv, int& i, const std::string& arg, const char* shortForm, std::string& out) {
    if (arg == shortForm) {
        if (i + 1 >= argc) return false;
        out = argv[++i];
        return true;
    }
    out = arg.substr(std::strlen(shortForm));
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return cudagfx::kExitUsageError;
    }

    cudagfx::CompilerOptions options;
    auto missing = [&](const std::string& opt) {
        std::cerr << "cudagfx: error: missing argument for " << opt << '\n';
        return cudagfx::kExitUsageError;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](std::string& dst) {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        std::string value;

        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return cudagfx::kExitOk;
        } else if (arg == "-o") {
            if (!next(options.outputHostPath)) return missing(arg);
        } else if (arg == "--kernels-header") {
            if (!next(options.outputKernelsHeaderPath)) return missing(arg);
        } else if (arg == "--shader-out") {
            if (!next(options.outputShaderPath)) return missing(arg);
        } else if (arg == "--shader-dir") {
            if (!next(options.outputShaderDir)) return missing(arg);
        } else if (arg == "--shaderc") {
            if (!next(options.shadercPath)) return missing(arg);
        } else if (arg == "--runtime-include") {
            if (!next(options.runtimeInclude)) return missing(arg);
        } else if (arg == "--depends") {
            if (!next(options.dependsPath)) return missing(arg);
        } else if (arg == "--target") {
            if (!next(value)) return missing(arg);
            cudagfx::ShaderTarget t;
            std::string error;
            if (!cudagfx::parseShaderTarget(value, t, error)) {
                std::cerr << "cudagfx: error: " << error << '\n';
                return cudagfx::kExitUsageError;
            }
            options.targets.push_back(t);
        } else if (arg == "--embed") {
            options.embed = true;
        } else if (arg == "--dump-ir") {
            options.dumpIr = true;
        } else if (arg == "--no-line-preserve") {
            options.noLinePreserve = true;
        } else if (arg.rfind("-I", 0) == 0) {
            if (!takeValue(argc, argv, i, arg, "-I", value)) return missing("-I");
            options.includePaths.push_back(value);
        } else if (arg.rfind("-i", 0) == 0) {
            if (!takeValue(argc, argv, i, arg, "-i", value)) return missing("-i");
            options.shadercIncludes.push_back(value);
        } else if (arg.rfind("-D", 0) == 0) {
            if (!takeValue(argc, argv, i, arg, "-D", value)) return missing("-D");
            const auto eq = value.find('=');
            if (eq == std::string::npos) options.defines[value] = "1";
            else options.defines[value.substr(0, eq)] = value.substr(eq + 1);
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "cudagfx: error: unknown option '" << arg << "' (see --help)\n";
            return cudagfx::kExitUsageError;
        } else if (options.inputFilePath.empty()) {
            options.inputFilePath = arg;
        } else {
            std::cerr << "cudagfx: error: only one input file is accepted ('" << options.inputFilePath
                      << "' and '" << arg << "')\n";
            return cudagfx::kExitUsageError;
        }
    }

    return cudagfx::Compiler().run(options);
}

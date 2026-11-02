// HostEmitter.cpp
#include "HostEmitter.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "CpuEmitter.hpp"
#include "KernelAbi.hpp"
#include "Literal.hpp"
#include "cudagfx_abi.h"

namespace cudagfx {

namespace {

struct Edit {
    std::size_t begin = 0;
    std::size_t end = 0;
    std::string text;
    std::size_t padding = 0; // newlines to append to keep the line numbering
};

bool isCudaHeader(const std::string& h) {
    static const char* const names[] = {"cuda_runtime.h", "cuda_runtime_api.h", "cuda.h", "vector_types.h",
                                        "vector_functions.h", "device_launch_parameters.h"};
    for (const char* n : names) {
        if (h == n) return true;
    }
    return false;
}

std::string bufferElementType(const Parameter& p) {
    Type elem = *p.type.inner;
    elem.isConst = false;
    return elem.toCppString();
}

const char* accessName(Parameter::Access a) {
    switch (a) {
        case Parameter::Access::ReadOnly: return "cudagfx::Access::Read";
        case Parameter::Access::WriteOnly: return "cudagfx::Access::Write";
        default: return "cudagfx::Access::ReadWrite";
    }
}

} // namespace

bool rewriteHostSource(const std::string& original, const ParsedModule& module, const Preprocessor& pp,
                       const HostRewriteConfig& config, DiagnosticsEngine& diags, std::string& out) {
    std::vector<Edit> edits;
    auto addEdit = [&](std::size_t begin, std::size_t end, std::string text) {
        if (begin > end || end > original.size()) return;
        Edit e;
        e.begin = begin;
        e.end = end;
        e.text = std::move(text);
        if (config.preserveLines) {
            e.padding = static_cast<std::size_t>(std::count(original.begin() + static_cast<std::ptrdiff_t>(begin),
                                                            original.begin() + static_cast<std::ptrdiff_t>(end), '\n'));
        }
        edits.push_back(std::move(e));
    };

    bool kernelsIncluded = false;
    for (const auto& inc : pp.mainIncludes()) {
        if (!isCudaHeader(inc.header)) continue;
        addEdit(inc.span.start.offset, inc.span.end.offset,
                kernelsIncluded ? std::string() : "#include \"" + config.kernelsHeaderInclude + "\"");
        kernelsIncluded = true;
    }

    for (const auto& fn : module.functions) {
        if (fn->span.start.file != module.mainFile) continue;
        if (!(fn->isKernel || (fn->isDevice && !fn->isHost))) continue;
        addEdit(fn->span.start.offset, fn->span.end.offset, "");
    }

    // A global constant used only by a kernel becomes unused once the kernel is removed from the host code:
    // clang (-Wunused-const-variable) would turn that into an error under -Werror. The C++17 attribute has no
    // effect if the host code uses the constant too.
    for (const GlobalConstDecl* g : module.usedGlobals) {
        if (g->declStart.file != module.mainFile) continue;
        addEdit(g->declStart.offset, g->declStart.offset, "[[maybe_unused]] ");
    }

    for (const auto& site : module.launchSites) {
        std::string target = site.templateBaseName.empty() ? site.kernelName : site.templateBaseName;
        std::string call = "cudagfx_gen::" + target + site.templateArgText + "::launch(" + site.gridExprText + ", " + site.blockExprText;
        for (const auto& a : site.argExprTexts) call += ", " + a;
        call += ")";
        addEdit(site.span.start.offset, site.span.end.offset, std::move(call));
    }

    std::stable_sort(edits.begin(), edits.end(), [](const Edit& a, const Edit& b) { return a.begin < b.begin; });

    std::string result;
    result.reserve(original.size() + 256);
    if (!kernelsIncluded) {
        result += "#include \"" + config.kernelsHeaderInclude + "\"\n";
        if (config.preserveLines) result += "#line 1 \"" + config.sourceName + "\"\n";
    }
    std::size_t pos = 0;
    for (const auto& e : edits) {
        if (e.begin < pos) {
            diags.reportError(DiagCode::CGFX0100, SourceLocation{}, "Host code rewrite: two modifications overlap (offset " + std::to_string(e.begin) + ")");
            return false;
        }
        result.append(original, pos, e.begin - pos);
        result += e.text;
        result.append(e.padding, '\n');
        pos = e.end;
    }
    result.append(original, pos, std::string::npos);
    if (!config.tailText.empty()) {
        if (result.empty() || result.back() != '\n') result += '\n';
        result += config.tailText;
    }
    out = std::move(result);
    return true;
}

std::string emitKernelsHeader(const ParsedModule& module, const VariantMap& variants,
                              const std::vector<ShaderArtifact>& artifacts, const KernelsHeaderConfig& config) {
    // Artifacts indexed by (kernel, variant): avoids a full scan per variant.
    std::unordered_map<std::string, std::vector<const ShaderArtifact*>> byVariant;
    for (const auto& a : artifacts) byVariant[a.kernel + "/" + a.variant.suffix()].push_back(&a);

    std::ostringstream os;
    os << "// Generated by CudaGFX from " << config.sourceName << " - do not edit by hand.\n";
    os << "#pragma once\n\n";
    os << "#include \"" << config.runtimeInclude << "\"\n\n";
    for (const auto& a : artifacts) os << "#include \"" << a.includePath << "\"\n";
    if (!artifacts.empty()) os << '\n';

    std::unordered_set<std::string> forwardDeclared;
    for (const auto& s : module.structs) {
        if (forwardDeclared.insert(s->name).second) {
            os << "struct " << s->name << ";\n";
        }
    }
    for (const StructDecl* s : module.usedStructs) {
        if (forwardDeclared.insert(s->name).second) {
            os << "struct " << s->name << ";\n";
        }
    }
    if (!forwardDeclared.empty()) os << '\n';

    os << "namespace cudagfx_gen {\n\n";

    if (config.emitCpu) os << emitCpuDeclarations(module);

    os << "namespace shaders {\n\n";
    for (const auto& k : module.functions) {
        if (!k->isKernel || !k->body) continue;
        for (const auto& v : variants.of(*k)) {
            os << "inline cudagfx::ShaderBinaryTable " << k->name << "_" << v.suffix() << "() {\n";
            os << "    cudagfx::ShaderBinaryTable table;\n";
            os << "    table.kernel = \"" << k->name << "\";\n";
            os << "    table.argsUniform = \"" << abi::argsUniformName(k->name) << "\";\n";
            os << "    table.gridUniform = \"" << abi::gridUniformName(k->name) << "\";\n";
            os << "    table.block[0] = " << v.x << "; table.block[1] = " << v.y << "; table.block[2] = " << v.z << ";\n";
            // The __constant__ objects the shader reads: the runtime uploads their current host content at each launch.
            const auto constants = uniformConstants(module, *k);
            if (!constants.empty()) {
                os << "    static const cudagfx::ConstantRef constants[] = {";
                for (std::size_t i = 0; i < constants.size(); ++i) {
                    const GlobalConstDecl* g = constants[i];
                    os << (i ? ", " : "") << "{\"" << g->name << "\", \"" << abi::constantUniformName(g->emitName) << "\", " << g->uniformSlots << ", "
                       << g->hostBytes << "}";
                }
                os << "};\n";
                os << "    table.constants = constants;\n    table.constantCount = " << constants.size() << ";\n";
            }
            const auto found = byVariant.find(k->name + "/" + v.suffix());
            if (found != byVariant.end()) {
                for (const ShaderArtifact* a : found->second) {
                    os << "    table." << a->backend << " = {" << a->symbol << ", sizeof(" << a->symbol << ")};\n";
                }
            }
            os << "    return table;\n}\n\n";
        }
    }
    os << "} // namespace shaders\n\n";

    // Primary templates for template kernels
    std::unordered_set<std::string> emittedTemplates;
    for (const auto& k : module.functions) {
        if (!k->isKernel || !k->body) continue;
        if (!k->templateBaseName.empty() && emittedTemplates.find(k->templateBaseName) == emittedTemplates.end()) {
            emittedTemplates.insert(k->templateBaseName);
            os << "template <";
            for (std::size_t i = 0; i < k->templateParams.size(); ++i) {
                if (i) os << ", ";
                const auto& tp = k->templateParams[i];
                if (tp.isType) os << "typename " << tp.name;
                else os << tp.typeName << " " << tp.name;
            }
            os << ">\nstruct " << k->templateBaseName << ";\n\n";
        }
    }

    for (const auto& k : module.functions) {
        if (!k->isKernel || !k->body) continue;
        const KernelBindings bindings = computeBindings(*k);
        std::string params;
        std::string fwdArgs;
        for (std::size_t i = 0; i < k->parameters.size(); ++i) {
            params += ", " + k->parameters[i].type.toCppString() + " p" + std::to_string(i);
            fwdArgs += (i ? ", p" : "p") + std::to_string(i);
        }
        if (k->templateBaseName.empty()) {
            os << "struct " << k->name << " {\n";
        } else {
            os << "template <>\nstruct " << k->templateBaseName << k->templateSpecialization << " {\n";
        }
        bool hasStructBuffer = false;
        for (const auto& p : k->parameters) {
            if (p.type.isPointer() && p.type.inner && p.type.inner->isStruct()) {
                hasStructBuffer = true;
                break;
            }
        }
        if (hasStructBuffer) {
            os << "    template <typename = void>\n";
        }
        os << "    static void launch(dim3 grid, dim3 blk" << params << ") {\n";
        std::string cpuFwdArgs;
        for (std::size_t i = 0; i < k->parameters.size(); ++i) {
            cpuFwdArgs += (i ? ", cudagfx::hostAddress(p" : "cudagfx::hostAddress(p") + std::to_string(i) + ")";
        }
        if (config.emitCpu) {
            os << "#if defined(CUDAGFX_CPU_ONLY)\n";
            os << "        cpu::" << k->name << "(grid, blk" << (cpuFwdArgs.empty() ? "" : ", " + cpuFwdArgs) << ");\n";
            os << "        return;\n";
            os << "#else\n";
            os << "        if (cudagfx::executionMode() == cudagfx::ExecutionMode::CPU) {\n";
            os << "            cpu::" << k->name << "(grid, blk" << (cpuFwdArgs.empty() ? "" : ", " + cpuFwdArgs) << ");\n";
            os << "            return;\n";
            os << "        }\n";
            os << "#endif\n";
        }
        for (const auto& v : variants.of(*k)) {
            os << "        if (blk.x == " << v.x << "u && blk.y == " << v.y << "u && blk.z == " << v.z << "u) {\n";
            os << "            cudagfx::Launch l(shaders::" << k->name << "_" << v.suffix() << "());\n";
            for (std::size_t i = 0; i < bindings.params.size(); ++i) {
                const ParamBinding& b = bindings.params[i];
                if (b.isBuffer) {
                    os << "            l.template buffer<" << bufferElementType(*b.param) << ">(" << b.bufferSlot << ", p" << i
                       << ", " << accessName(b.param->access) << ");\n";
                } else {
                    os << "            l.arg(p" << i << ");\n";
                }
            }
            os << "            l.dispatch(grid, blk);\n";
            os << "            return;\n";
            os << "        }\n";
        }
        os << "        cudagfx::detail::badBlockSize(\"" << (k->templateBaseName.empty() ? k->name : k->templateBaseName) << "\", blk);\n";
        os << "    }\n};\n\n";
    }
    os << "} // namespace cudagfx_gen\n";
    return os.str();
}

} // namespace cudagfx


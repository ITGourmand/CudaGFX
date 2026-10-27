// Compiler.cpp: orchestration. Each step (read, analyze, shaders, shaderc, host, dependencies)
// is a short method of Session; none of them writes a file before the analysis has succeeded.
#include "Compiler.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "Diagnostics.hpp"
#include "HostEmitter.hpp"
#include "KernelAbi.hpp"
#include "Parser.hpp"
#include "Preprocessor.hpp"
#include "Sema.hpp"
#include "ShaderEmitter.hpp"
#include "ShadercDriver.hpp"
#include "SourceManager.hpp"
#include "Variants.hpp"

namespace cudagfx {

namespace {

namespace fs = std::filesystem;

std::string baseName(const std::string& path) { return fs::path(path).filename().string(); }

fs::path absoluteNormal(const std::string& path) {
    std::error_code ec;
    const fs::path abs = fs::absolute(fs::path(path), ec);
    return (ec ? fs::path(path) : abs).lexically_normal();
}

// Path of `target` relative to the directory `fromDir`, in portable form (forward slashes).
std::string relativeTo(const std::string& target, const fs::path& fromDir) {
    const fs::path rel = absoluteNormal(target).lexically_relative(absoluteNormal(fromDir.empty() ? "." : fromDir.string()));
    return (rel.empty() ? fs::path(target) : rel).generic_string();
}

bool writeIfChanged(const std::string& path, const std::string& content, std::string& error) {
    std::error_code ec;
    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent, ec);
    {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            std::ostringstream ss;
            ss << in.rdbuf();
            if (ss.str() == content) return true; // unchanged: the timestamp is left alone (incremental builds)
        }
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        error = "Unable to write '" + path + "'";
        return false;
    }
    out << content;
    out.flush();
    if (!out) {
        error = "Write error in '" + path + "'";
        return false;
    }
    return true;
}

std::string jsonEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            default: o += c;
        }
    }
    return o;
}

std::string depEscape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == ' ' || c == '#') o += '\\';
        o += c;
    }
    return o;
}

struct ShaderFile {
    const FunctionDecl* kernel = nullptr;
    BlockVariant variant;
    std::string scPath;
};

class Session {
public:
    Session(const CompilerOptions& options, std::ostream& out, std::ostream& err)
        : opt_(options), out_(out), err_(err), diags_(&sources_) {}

    int run() {
        if (opt_.inputFilePath.empty()) return usage("no input file");
        if (opt_.embed && opt_.shadercPath.empty()) return usage("--embed requires --shaderc");
        if (!opt_.shadercPath.empty() && opt_.targets.empty()) opt_.targets = defaultShaderTargets();

        if (int rc = loadSource()) return rc;
        analyze();
        diags_.printAll(err_);
        if (diags_.hasErrors()) return kExitSourceError;

        const auto kernels = kernelList();
        variants_ = std::make_unique<VariantMap>(module_);
        if (opt_.dumpIr) dumpIr(kernels);

        std::vector<ShaderFile> shaders;
        if (int rc = emitShaders(kernels, shaders)) return rc;
        std::vector<ShaderArtifact> artifacts;
        if (int rc = runShaderc(shaders, artifacts)) return rc;
        if (int rc = emitHost(artifacts)) return rc;
        if (int rc = writeDepends(shaders)) return rc;
        return kExitOk;
    }

private:
    int usage(const std::string& why) {
        err_ << "cudagfx: error: " << why << '\n';
        return kExitUsageError;
    }

    int loadSource() {
        std::ifstream in(opt_.inputFilePath, std::ios::binary);
        if (!in) {
            err_ << "cudagfx: error: unable to open '" << opt_.inputFilePath << "'\n";
            return kExitUsageError;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        mainFile_ = sources_.add(opt_.inputFilePath, ss.str());
        return 0;
    }

    void analyze() {
        pp_ = std::make_unique<Preprocessor>(diags_, sources_, opt_.includePaths);
        for (const auto& d : opt_.defines) pp_->defineMacro(d.first, d.second);
        std::vector<Token> tokens = pp_->process(mainFile_);
        if (!diags_.hasErrors()) {
            Parser parser(*pp_, sources_, std::move(tokens), mainFile_, diags_);
            module_ = parser.parse();
        }
        if (diags_.hasErrors()) return;

        Sema(diags_).analyze(module_);

        // A kernel defined in a header would be neither extracted nor removed from the host code: reject it if it is launched.
        for (const auto& f : module_.functions) {
            if (!f->isKernel || !f->body || f->span.start.file == mainFile_) continue;
            for (const auto& site : module_.launchSites) {
                if (site.kernelName == f->name) {
                    diags_.reportError(DiagCode::CGFX0019, f->loc,
                                       "Kernel '" + f->name + "' is defined in an included file: define it in the main file");
                    break;
                }
            }
        }

        // Output consistency: a --shader-out path without a pattern can only designate one kernel.
        const auto kernels = kernelList();
        if (!opt_.outputShaderPath.empty() && opt_.outputShaderPath.find('%') == std::string::npos && kernels.size() > 1) {
            std::string names;
            for (const auto* k : kernels) names += (names.empty() ? "" : ", ") + k->name;
            diags_.reportError(DiagCode::CGFX0011, kernels[1]->loc,
                               "Several kernels (" + names + "): --shader-out '" + opt_.outputShaderPath +
                                   "' is ambiguous, use a pattern with '%' (for example cs_%.sc)");
        }
        // ABI binding: only if the rest is valid (otherwise an already rejected type would produce a spurious error).
        if (!diags_.hasErrors()) {
            for (const auto* k : kernels) {
                const KernelBindings b = computeBindings(*k);
                if (!b.ok) diags_.reportError(DiagCode::CGFX0008, k->loc, b.error);
            }
        }
    }

    std::vector<const FunctionDecl*> kernelList() const {
        std::vector<const FunctionDecl*> out;
        for (const auto& f : module_.functions) {
            if (f->isKernel && f->body && f->span.start.file == mainFile_) out.push_back(f.get());
        }
        return out;
    }

    std::string scPathFor(const FunctionDecl& k, const BlockVariant& v, std::size_t variantCount) const {
        if (!opt_.outputShaderPath.empty()) {
            std::string p = opt_.outputShaderPath;
            for (std::size_t pos = p.find('%'); pos != std::string::npos; pos = p.find('%', pos + k.name.size())) {
                p.replace(pos, 1, k.name);
            }
            if (variantCount > 1) {
                const std::string suffix = "_" + v.suffix();
                const std::string ext = ".sc";
                if (p.size() >= ext.size() && p.compare(p.size() - ext.size(), ext.size(), ext) == 0) {
                    p.insert(p.size() - ext.size(), suffix);
                } else {
                    p += suffix;
                }
            }
            return p;
        }
        const fs::path dir = opt_.outputShaderDir.empty() ? fs::path(".") : fs::path(opt_.outputShaderDir);
        return (dir / ("cs_" + k.name + "_" + v.suffix() + ".sc")).string();
    }

    int emitShaders(const std::vector<const FunctionDecl*>& kernels, std::vector<ShaderFile>& shaders) {
        ShaderEmitter emitter;
        for (const FunctionDecl* k : kernels) {
            const auto variants = variants_->of(*k);
            for (const auto& v : variants) {
                ShaderFile sf;
                sf.kernel = k;
                sf.variant = v;
                sf.scPath = scPathFor(*k, v, variants.size());
                const std::string text = emitter.emit(module_, *k, v, baseName(opt_.inputFilePath));
                std::string error;
                if (!writeIfChanged(sf.scPath, text, error)) {
                    err_ << "cudagfx: error: " << error << '\n';
                    return kExitInternalError;
                }
                shaders.push_back(std::move(sf));
            }
        }
        return 0;
    }

    int runShaderc(const std::vector<ShaderFile>& shaders, std::vector<ShaderArtifact>& artifacts) {
        if (opt_.shadercPath.empty()) return 0;
        ShadercDriver driver(opt_.shadercPath, opt_.shadercIncludes);
        for (const auto& sf : shaders) {
            const std::string base = fs::path(sf.scPath).replace_extension("").string();
            for (const auto& target : opt_.targets) {
                const std::string symbol = shaderSymbolName(sf.kernel->name, sf.variant, target.backend);
                const std::string outPath = base + "_" + target.backend + (opt_.embed ? ".h" : ".bin");
                std::string error;
                if (!driver.compile(sf.scPath, outPath, target, opt_.embed ? symbol : std::string(), error)) {
                    err_ << "cudagfx: error: " << error << '\n';
                    return kExitShadercFailure;
                }
                if (opt_.embed) {
                    ShaderArtifact a;
                    a.kernel = sf.kernel->name;
                    a.variant = sf.variant;
                    a.backend = target.backend;
                    a.symbol = symbol;
                    a.includePath = outPath; // made relative to the stubs header at emission time
                    artifacts.push_back(std::move(a));
                }
            }
        }
        return 0;
    }

    int emitHost(std::vector<ShaderArtifact>& artifacts) {
        if (opt_.outputHostPath.empty() && opt_.outputKernelsHeaderPath.empty()) return 0;

        fs::path hostDir = fs::path(opt_.outputHostPath).parent_path();
        std::string headerPath = opt_.outputKernelsHeaderPath;
        if (headerPath.empty()) {
            headerPath = (hostDir / (fs::path(opt_.outputHostPath).stem().string() + "_kernels.h")).string();
        }
        const fs::path headerDir = fs::path(headerPath).parent_path();
        for (auto& a : artifacts) a.includePath = relativeTo(a.includePath, headerDir);

        KernelsHeaderConfig hc;
        hc.sourceName = baseName(opt_.inputFilePath);
        hc.runtimeInclude = opt_.runtimeInclude;
        std::string error;
        if (!writeIfChanged(headerPath, emitKernelsHeader(module_, *variants_, artifacts, hc), error)) {
            err_ << "cudagfx: error: " << error << '\n';
            return kExitInternalError;
        }

        if (opt_.outputHostPath.empty()) return 0;
        HostRewriteConfig rc;
        rc.kernelsHeaderInclude = relativeTo(headerPath, hostDir);
        rc.sourceName = baseName(opt_.inputFilePath);
        rc.preserveLines = !opt_.noLinePreserve;
        std::string host;
        if (!rewriteHostSource(sources_.content(mainFile_), module_, *pp_, rc, diags_, host)) {
            diags_.printAll(err_);
            return kExitInternalError;
        }
        if (!writeIfChanged(opt_.outputHostPath, host, error)) {
            err_ << "cudagfx: error: " << error << '\n';
            return kExitInternalError;
        }
        return 0;
    }

    int writeDepends(const std::vector<ShaderFile>& shaders) {
        if (opt_.dependsPath.empty()) return 0;
        std::string target = !opt_.outputHostPath.empty() ? opt_.outputHostPath
                             : (!shaders.empty() ? shaders.front().scPath : opt_.dependsPath);
        std::string text = depEscape(target) + ": " + depEscape(opt_.inputFilePath);
        for (FileId id : pp_->includedFiles()) text += " " + depEscape(sources_.path(id));
        text += "\n";
        std::string error;
        if (!writeIfChanged(opt_.dependsPath, text, error)) {
            err_ << "cudagfx: error: " << error << '\n';
            return kExitInternalError;
        }
        return 0;
    }

    void dumpIr(const std::vector<const FunctionDecl*>& kernels) const {
        out_ << "{\n  \"source\": \"" << jsonEscape(opt_.inputFilePath) << "\",\n  \"kernels\": [";
        for (std::size_t i = 0; i < kernels.size(); ++i) {
            const FunctionDecl& k = *kernels[i];
            out_ << (i ? ",\n" : "\n") << "    {\"name\": \"" << jsonEscape(k.name) << "\", \"params\": [";
            for (std::size_t p = 0; p < k.parameters.size(); ++p) {
                const Parameter& prm = k.parameters[p];
                out_ << (p ? ", " : "") << "{\"name\": \"" << jsonEscape(prm.name) << "\", \"type\": \""
                     << jsonEscape(prm.type.toCppString()) << "\"}";
            }
            out_ << "], \"variants\": [";
            const auto vs = variants_->of(k);
            for (std::size_t v = 0; v < vs.size(); ++v) {
                out_ << (v ? ", " : "") << "[" << vs[v].x << ", " << vs[v].y << ", " << vs[v].z << "]";
            }
            out_ << "], \"helpers\": [";
            const auto it = module_.helperClosure.find(&k);
            if (it != module_.helperClosure.end()) {
                for (std::size_t h = 0; h < it->second.size(); ++h) {
                    out_ << (h ? ", " : "") << "\"" << jsonEscape(it->second[h]->name) << "\"";
                }
            }
            out_ << "], \"shared_bytes\": " << k.sharedBytes << ", \"uses_grid_dim\": " << (k.usesGridDim ? "true" : "false") << "}";
        }
        out_ << "\n  ],\n  \"launch_sites\": [";
        for (std::size_t i = 0; i < module_.launchSites.size(); ++i) {
            const auto& s = module_.launchSites[i];
            out_ << (i ? ",\n" : "\n") << "    {\"kernel\": \"" << jsonEscape(s.kernelName) << "\", \"line\": " << s.kernelLoc.line
                 << ", \"block\": [" << s.evaluatedBlock[0] << ", " << s.evaluatedBlock[1] << ", " << s.evaluatedBlock[2] << "]}";
        }
        out_ << "\n  ]\n}\n";
    }

    CompilerOptions opt_;
    std::ostream& out_;
    std::ostream& err_;
    SourceManager sources_;
    DiagnosticsEngine diags_;
    std::unique_ptr<Preprocessor> pp_;
    ParsedModule module_;
    std::unique_ptr<VariantMap> variants_;
    FileId mainFile_ = kNoFile;
};

} // namespace

int Compiler::run(const CompilerOptions& options, std::ostream& out, std::ostream& err) const {
    try {
        return Session(options, out, err).run();
    } catch (const std::exception& e) {
        err << "cudagfx: internal error: CGFX0100: " << e.what() << '\n';
        return kExitInternalError;
    } catch (...) {
        err << "cudagfx: internal error: CGFX0100: unknown exception\n";
        return kExitInternalError;
    }
}

} // namespace cudagfx

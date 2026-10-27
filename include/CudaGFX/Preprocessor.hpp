// Preprocessor.hpp: the CudaGFX mini-preprocessor.
//
// Role: produce, for the analysis, the ACTIVE token stream of the main file (conditionals resolved,
// #include "x.h" followed, directives removed). Macros are NOT expanded in this stream: the host code is
// rewritten from the original text and must keep its macros intact. Expansion (object-like, function-like,
// variadic) is provided on demand by expand(), for device code and for constant evaluation.
//
//
// Not supported (CGFX0014): the # and ## operators in a macro body, #include_next, an active #error.
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "Diagnostics.hpp"
#include "SourceManager.hpp"
#include "Token.hpp"

namespace cudagfx {

struct MacroDefinition {
    std::string name;
    bool functionLike = false;
    bool variadic = false;
    std::vector<std::string> params;
    std::vector<Token> body;
    SourceLocation location;
};

// An #include directive found in the main file (used when rewriting the host code).
struct IncludeDirective {
    SourceSpan span;     // from the '#' to the end of the header name
    std::string header;  // name without delimiters
    bool angled = false;
};

class Preprocessor {
public:
    Preprocessor(DiagnosticsEngine& diags, SourceManager& sources, std::vector<std::string> includePaths = {});

    // -D<name>[=<value>]: `valueText` is parsed as C code (for example "2*64").
    void defineMacro(const std::string& name, const std::string& valueText);

    // Active tokens of the main file (and of the #include "..." files it follows), terminated by EndOfFile.
    std::vector<Token> process(FileId mainFile);

    // Full macro expansion; the produced tokens carry the location of the invocation.
    std::vector<Token> expand(const std::vector<Token>& tokens) const;

    bool isDefined(const std::string& name) const { return macros_.count(name) != 0; }

    const std::vector<IncludeDirective>& mainIncludes() const { return mainIncludes_; }
    // User headers that were actually followed (main file excluded), for --depends.
    const std::vector<FileId>& includedFiles() const { return includedFiles_; }

private:
    struct Conditional {
        bool parentActive = true;
        bool taken = false;
        bool active = false;
        bool sawElse = false;
        SourceLocation location;
    };

    void processFile(FileId file, std::vector<Token>& out, unsigned depth);
    std::size_t directiveEnd(const std::vector<Token>& toks, std::size_t i) const;
    void handleDefine(const std::vector<Token>& toks, std::size_t first, std::size_t last);
    void handleInclude(FileId includer, const std::vector<Token>& toks, std::size_t dirIdx, std::size_t last,
                       std::vector<Token>& out, unsigned depth);
    bool evaluateCondition(const std::vector<Token>& toks, std::size_t first, std::size_t last,
                           const SourceLocation& at, bool& result);
    std::string resolveInclude(const std::string& header, FileId includer) const;

    void expandInto(const std::vector<Token>& in, std::size_t begin, std::size_t end, std::vector<Token>& out,
                    std::vector<const MacroDefinition*>& active, std::size_t& budget) const;

    DiagnosticsEngine& diags_;
    SourceManager& sources_;
    std::vector<std::string> includePaths_;
    std::unordered_map<std::string, MacroDefinition> macros_;
    std::unordered_set<FileId> pragmaOnce_;
    std::vector<IncludeDirective> mainIncludes_;
    std::vector<FileId> includedFiles_;
    FileId mainFile_ = kNoFile;
    FileId commandLineFile_ = kNoFile;
};

} // namespace cudagfx

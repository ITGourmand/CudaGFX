// Diagnostics.hpp: the CudaGFX diagnostics engine.
//
// Every error or warning meant for the user goes through this engine (never through scattered
// writes to std::cerr); CGFX0100 is reserved for genuine internal errors.
#pragma once

#include <iosfwd>
#include <string>
#include <unordered_set>
#include <vector>
#include "SourceLocation.hpp"

namespace cudagfx {

class SourceManager;

enum class DiagLevel {
    Note,
    Warning,
    Error
};

enum class DiagCode {
    CGFX0001, // Pointer arithmetic / aliasing / dereference not supported in a kernel
    CGFX0002, // Unsupported type (double, 64-bit integer, char, struct parameter...)
    CGFX0003, // Unsupported warp intrinsic
    CGFX0004, // extern __shared__ / dynamic shared memory not supported (a static size is required)
    CGFX0005, // Function not supported in a kernel (printf, malloc, recursion...)
    CGFX0006, // Block size is not a compile-time constant at the launch site
    CGFX0007, // Block size inconsistent with __launch_bounds__ (warning)
    CGFX0008, // Block size or shared memory size beyond the limits
    CGFX0009, // C++ construct not supported in a kernel (lambda, virtual, exception, template...)
    CGFX0010, // Unknown identifier, or __device__ function without a visible definition
    CGFX0011, // Ambiguous kernel name for --shader-out
    CGFX0012, // Unsupported buffer element type (float3*, double*, struct*...)
    CGFX0013, // Kernel launch <<<>>> inside a macro or an included file, or malformed
    CGFX0014, // Unsupported or malformed preprocessor directive
    CGFX0015, // Invalid atomic, or its result is used in the middle of an expression
    CGFX0016, // Reserved identifier (bgfx_*, gl_*) or name collision
    CGFX0017, // Warning: % or / on signed integers
    CGFX0018, // Lexical or syntax error in the source
    CGFX0019, // Type or semantic error (invalid operands, write to a const...)
    CGFX0020, // Warning: behaviour differs between CUDA and the shader (fmodf...)
    CGFX0100  // Internal error
};

struct Diagnostic {
    DiagLevel level;
    DiagCode code;
    SourceLocation location;
    std::string message;
};

class DiagnosticsEngine {
public:
    explicit DiagnosticsEngine(const SourceManager* sources = nullptr) : sources_(sources) {}

    void setSourceManager(const SourceManager* sources) { sources_ = sources; }

    // An identical diagnostic (same code, level and location) is recorded only once.
    void report(DiagLevel level, DiagCode code, SourceLocation loc, std::string msg);
    void reportError(DiagCode code, SourceLocation loc, std::string msg) {
        report(DiagLevel::Error, code, loc, std::move(msg));
    }
    void reportWarning(DiagCode code, SourceLocation loc, std::string msg) {
        report(DiagLevel::Warning, code, loc, std::move(msg));
    }

    // While active, diagnostics are counted but not recorded
    // (speculative analysis, e.g. the initializer of a host-side global constant).
    void setMuted(bool muted) { muted_ = muted; }
    bool isMuted() const { return muted_; }
    // Number of errors swallowed since the last resetMutedErrors().
    std::size_t mutedErrorCount() const { return mutedErrors_; }
    void resetMutedErrors() { mutedErrors_ = 0; }

    bool hasErrors() const { return errorCount_ > 0; }
    std::size_t errorCount() const { return errorCount_; }
    std::size_t warningCount() const { return warningCount_; }
    const std::vector<Diagnostic>& diagnostics() const { return diags_; }

    // Writes every diagnostic: `file:line:col: error: CGFXnnnn: message`, followed by the offending source
    // line (from the right file, included headers too) and a caret.
    void printAll(std::ostream& os) const;

    static const char* codeToString(DiagCode code);
    static const char* levelToString(DiagLevel level);

    static constexpr std::size_t kMaxErrors = 100;

private:
    const SourceManager* sources_;
    std::vector<Diagnostic> diags_;
    std::unordered_set<std::string> seen_;
    std::size_t errorCount_ = 0;
    std::size_t warningCount_ = 0;
    std::size_t mutedErrors_ = 0;
    bool muted_ = false;
    bool truncated_ = false;
};

} // namespace cudagfx

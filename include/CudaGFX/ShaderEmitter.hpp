// ShaderEmitter.hpp: generation of the bgfx compute source (.sc) of a kernel.
//
// The emitter makes NO typing decision: Sema has already made every conversion explicit
// (CastExpr) and fixed the emission names. It only translates the analyzed AST to text.
#pragma once

#include <string>
#include "AST.hpp"
#include "Variants.hpp"

namespace cudagfx {

class ShaderEmitter {
public:
    // `sourceName` only appears in the header comment (base name of the file, so that the
    // output is reproducible whatever the working directory).
    std::string emit(const ParsedModule& module, const FunctionDecl& kernel, const BlockVariant& block,
                     const std::string& sourceName) const;
};

} // namespace cudagfx

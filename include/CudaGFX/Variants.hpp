// Variants.hpp: block sizes (shader variants) of a kernel.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "AST.hpp"

namespace cudagfx {

struct BlockVariant {
    unsigned x = 256, y = 1, z = 1;

    unsigned long long threads() const { return static_cast<unsigned long long>(x) * y * z; }
    // "256", "16x16", "8x8x4": suffix of the shader names and of the stub functions.
    std::string suffix() const;
    bool operator==(const BlockVariant& o) const { return x == o.x && y == o.y && z == o.z; }
};

// Index of the variants of every kernel, built in a single pass over the launch sites
// (variantsOf() rescans every site: avoid it inside a loop over kernels).
class VariantMap {
public:
    explicit VariantMap(const ParsedModule& module);
    // Variants of the kernel (defaults if it is never launched), in order of appearance.
    std::vector<BlockVariant> of(const FunctionDecl& kernel) const;

private:
    std::unordered_map<std::string, std::vector<BlockVariant>> launched_;
};

// One variant per distinct block size seen at the kernel's launch sites (in order of
// appearance). Without any launch site: __launch_bounds__ if present, otherwise 256 threads.
std::vector<BlockVariant> variantsOf(const ParsedModule& module, const FunctionDecl& kernel);

} // namespace cudagfx

// Variants.cpp
#include "Variants.hpp"

#include <algorithm>

namespace cudagfx {

std::string BlockVariant::suffix() const {
    if (y == 1 && z == 1) return std::to_string(x);
    if (z == 1) return std::to_string(x) + "x" + std::to_string(y);
    return std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z);
}

namespace {

BlockVariant variantOfSite(const LaunchSite& site) {
    BlockVariant v;
    v.x = site.evaluatedBlock[0];
    v.y = site.evaluatedBlock[1];
    v.z = site.evaluatedBlock[2];
    return v;
}

void addUnique(std::vector<BlockVariant>& out, const BlockVariant& v) {
    if (std::find(out.begin(), out.end(), v) == out.end()) out.push_back(v);
}

std::vector<BlockVariant> withDefault(std::vector<BlockVariant> out, const FunctionDecl& kernel) {
    if (out.empty()) {
        BlockVariant v;
        v.x = kernel.launchBounds != 0 ? kernel.launchBounds : 256;
        out.push_back(v);
    }
    return out;
}

} // namespace

VariantMap::VariantMap(const ParsedModule& module) {
    for (const auto& site : module.launchSites) addUnique(launched_[site.kernelName], variantOfSite(site));
}

std::vector<BlockVariant> VariantMap::of(const FunctionDecl& kernel) const {
    const auto it = launched_.find(kernel.name);
    return withDefault(it == launched_.end() ? std::vector<BlockVariant>{} : it->second, kernel);
}

std::vector<BlockVariant> variantsOf(const ParsedModule& module, const FunctionDecl& kernel) {
    std::vector<BlockVariant> out;
    for (const auto& site : module.launchSites) {
        if (site.kernelName == kernel.name) addUnique(out, variantOfSite(site));
    }
    return withDefault(std::move(out), kernel);
}

} // namespace cudagfx

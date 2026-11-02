// KernelAbi.cpp
#include "KernelAbi.hpp"

#include <algorithm>

#include "cudagfx_abi.h"

namespace cudagfx {

KernelBindings computeBindings(const FunctionDecl& kernel) {
    KernelBindings out;
    abi::ArgLayout layout;
    for (const Parameter& p : kernel.parameters) {
        ParamBinding b;
        b.param = &p;
        if (p.type.isPointer()) {
            b.isBuffer = true;
            b.bufferSlot = out.bufferCount++;
        } else {
            b.components = p.type.componentCount();
            if (!layout.place(b.components, b.lane)) {
                out.ok = false;
                out.error = "The by-value parameters of kernel '" + kernel.name + "' exceed " +
                            std::to_string(abi::kMaxArgLanes) + " 32-bit words (" + std::to_string(abi::kMaxArgSlots) + " vec4)";
                return out;
            }
        }
        out.params.push_back(b);
    }
    if (out.bufferCount > kMaxBufferBindings) {
        out.ok = false;
        out.error = "Kernel '" + kernel.name + "' has " + std::to_string(out.bufferCount) + " buffers (maximum: " +
                    std::to_string(kMaxBufferBindings) + ")";
    }
    out.argSlots = layout.slotCount();
    return out;
}

unsigned constantElementBytes(const Type& element) {
    switch (element.kind) {
        case Type::Kind::Int:
        case Type::Kind::UInt:
        case Type::Kind::Float: return 4;
        default: break;
    }
    if (element.isVector()) return 4 * element.componentCount();
    return 0;
}

std::vector<const GlobalConstDecl*> uniformConstants(const ParsedModule& module, const FunctionDecl& kernel) {
    std::vector<const GlobalConstDecl*> out;
    auto add = [&out](const FunctionDecl& fn) {
        for (const GlobalConstDecl* g : fn.usedConstants) {
            if (g->uniformBacked && std::find(out.begin(), out.end(), g) == out.end()) out.push_back(g);
        }
    };
    add(kernel);
    const auto it = module.helperClosure.find(&kernel);
    if (it != module.helperClosure.end()) {
        for (const FunctionDecl* h : it->second) add(*h);
    }
    return out;
}

} // namespace cudagfx

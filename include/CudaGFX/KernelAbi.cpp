// KernelAbi.cpp
#include "KernelAbi.hpp"

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

} // namespace cudagfx

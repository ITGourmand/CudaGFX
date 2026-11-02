// StructLayout.cpp
#include "StructLayout.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace cudagfx {

static StructLayout computeLayoutDepth(const StructDecl& decl, LayoutRules rules, const StructLookup& lookup, int depth);

namespace {

unsigned alignUp(unsigned v, unsigned a) { return (v + a - 1) / a * a; }

struct MemberShape {
    unsigned size = 0;
    unsigned align = 4;
    bool valid = true;
    std::string problem;
};

MemberShape shapeOf(const Type& t, LayoutRules rules, const StructLookup& lookup, int depth) {
    MemberShape m;
    switch (t.kind) {
        case Type::Kind::Int:
        case Type::Kind::UInt:
        case Type::Kind::Float:
            m.size = 4;
            m.align = 4;
            return m;
        case Type::Kind::Bool:
            m.valid = false;
            m.problem = "a bool field has no portable layout (1 byte on the host, 4 in the shader): use int";
            return m;
        case Type::Kind::Float2:
        case Type::Kind::Int2:
        case Type::Kind::UInt2:
            m.size = 8;
            m.align = rules == LayoutRules::Tight ? 4 : 8;
            return m;
        case Type::Kind::Float4:
        case Type::Kind::Int4:
        case Type::Kind::UInt4:
            m.size = 16;
            m.align = rules == LayoutRules::Tight ? 4 : 16;
            return m;
        case Type::Kind::Float3:
        case Type::Kind::Int3:
        case Type::Kind::UInt3:
            m.valid = false;
            m.problem = "a 3-component vector field has no portable layout (its size is 12 or 16 bytes depending on the backend): use the 4-component type";
            return m;
        case Type::Kind::Struct: {
            const StructDecl* nested = depth < 16 ? lookup(t.name) : nullptr;
            if (!nested) {
                m.valid = false;
                m.problem = "unknown or recursive struct '" + t.name + "'";
                return m;
            }
            const StructLayout inner = computeLayoutDepth(*nested, rules, lookup, depth + 1);
            m.size = inner.size;
            m.align = inner.align;
            m.valid = inner.valid;
            m.problem = inner.problem;
            return m;
        }
        default:
            m.valid = false;
            m.problem = "this field type has no portable layout in a buffer";
            return m;
    }
}

} // namespace

static StructLayout computeLayoutDepth(const StructDecl& decl, LayoutRules rules, const StructLookup& lookup, int depth) {
    StructLayout out;
    unsigned offset = 0;
    unsigned maxAlign = 4;
    for (const auto& f : decl.fields) {
        const MemberShape m = shapeOf(f.type, rules, lookup, depth);
        if (!m.valid) {
            out.valid = false;
            if (out.problem.empty()) out.problem = "field '" + f.name + "': " + m.problem;
        }
        offset = alignUp(offset, m.align);
        out.offsets.push_back(offset);
        offset += m.size;
        maxAlign = std::max(maxAlign, m.align);
    }
    out.align = maxAlign;
    // C++ and std430 round the size up to the alignment; HLSL structured buffers only keep 4-byte granularity.
    out.size = alignUp(offset, rules == LayoutRules::Tight ? 4u : maxAlign);
    return out;
}

StructLayout computeLayout(const StructDecl& decl, LayoutRules rules, const StructLookup& lookup) {
    return computeLayoutDepth(decl, rules, lookup, 0);
}

namespace {

std::string check(const StructDecl& decl, const StructLookup& lookup, unsigned maxBytes, std::unordered_set<std::string>& seen) {
    if (!seen.insert(decl.name).second) return {};
    const StructLayout host = computeLayout(decl, LayoutRules::Host, lookup);
    if (!host.valid) return "struct '" + decl.name + "': " + host.problem;
    const StructLayout std430 = computeLayout(decl, LayoutRules::Std430, lookup);
    const StructLayout tight = computeLayout(decl, LayoutRules::Tight, lookup);

    for (std::size_t i = 0; i < decl.fields.size(); ++i) {
        if (host.offsets[i] != std430.offsets[i] || host.offsets[i] != tight.offsets[i]) {
            std::ostringstream os;
            os << "struct '" << decl.name << "': field '" << decl.fields[i].name << "' is at offset " << host.offsets[i]
               << " on the host (C++), " << std430.offsets[i] << " in std430 (OpenGL/Vulkan/WebGPU) and " << tight.offsets[i]
               << " in HLSL (Direct3D); insert explicit padding fields so that every 2-component vector is 8-byte aligned and "
                  "every 4-component vector 16-byte aligned";
            return os.str();
        }
    }
    if (host.size != std430.size || host.size != tight.size) {
        std::ostringstream os;
        os << "struct '" << decl.name << "' is " << host.size << " bytes on the host (C++), " << std430.size << " in std430 and "
           << tight.size << " in HLSL; add trailing padding so that its size is a multiple of its largest alignment (" << host.align << ")";
        return os.str();
    }
    if (host.size > maxBytes) {
        return "struct '" + decl.name + "' is " + std::to_string(host.size) + " bytes, beyond the limit of " + std::to_string(maxBytes) +
               " bytes for a buffer element";
    }
    for (const auto& f : decl.fields) {
        if (f.type.isStruct()) {
            if (const StructDecl* nested = lookup(f.type.name)) {
                std::string p = check(*nested, lookup, maxBytes, seen);
                if (!p.empty()) return p;
            }
        }
    }
    return {};
}

} // namespace

std::string portableLayoutProblem(const StructDecl& decl, const StructLookup& lookup, unsigned maxBytes) {
    std::unordered_set<std::string> seen;
    return check(decl, lookup, maxBytes, seen);
}

std::string emitHostLayoutAsserts(const std::vector<const StructDecl*>& structs, const StructLookup& lookup) {
    if (structs.empty()) return {};
    std::ostringstream os;
    os << "\n// CudaGFX: layout of the structs used as buffer elements (generated; the shaders assume exactly this).\n";
    os << "#include <cstddef>\n";
    for (const StructDecl* s : structs) {
        const StructLayout host = computeLayout(*s, LayoutRules::Host, lookup);
        os << "static_assert(sizeof(" << s->name << ") == " << host.size << " && alignof(" << s->name << ") == " << host.align
           << ", \"CudaGFX: size or alignment of struct " << s->name << " differs from the layout of the generated shaders\");\n";
        for (std::size_t i = 0; i < s->fields.size(); ++i) {
            os << "static_assert(offsetof(" << s->name << ", " << s->fields[i].name << ") == " << host.offsets[i]
               << ", \"CudaGFX: offset of " << s->name << "::" << s->fields[i].name << " differs from the layout of the generated shaders\");\n";
        }
    }
    return os.str();
}

} // namespace cudagfx

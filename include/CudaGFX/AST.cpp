// AST.cpp
#include "AST.hpp"

namespace cudagfx {

Type Type::scalar(Kind k) {
    Type t;
    t.kind = k;
    return t;
}

Type Type::makeVector(Kind scalarKind, unsigned components) {
    Kind k = Kind::Unknown;
    if (components >= 2 && components <= 4) {
        const unsigned n = components - 2;
        switch (scalarKind) {
            case Kind::Float: k = static_cast<Kind>(static_cast<int>(Kind::Float2) + static_cast<int>(n)); break;
            case Kind::Int: k = static_cast<Kind>(static_cast<int>(Kind::Int2) + static_cast<int>(n)); break;
            case Kind::UInt: k = static_cast<Kind>(static_cast<int>(Kind::UInt2) + static_cast<int>(n)); break;
            default: break;
        }
    } else if (components == 1 && (scalarKind == Kind::Float || scalarKind == Kind::Int || scalarKind == Kind::UInt)) {
        k = scalarKind;
    }
    return scalar(k);
}

Type Type::makePointer(const Type& pointee, bool pointeeConst) {
    Type t;
    t.kind = Kind::Pointer;
    t.inner = std::make_shared<const Type>(pointee);
    t.isConst = pointeeConst || pointee.isConst;
    return t;
}

Type Type::makeArray(const Type& element, unsigned size) {
    Type t;
    t.kind = Kind::Array;
    t.inner = std::make_shared<const Type>(element);
    t.arraySize = size;
    return t;
}

Type Type::makeStruct(std::string emitName) {
    Type t;
    t.kind = Kind::Struct;
    t.name = std::move(emitName);
    return t;
}

Type Type::makeUnsupported(std::string cppName) {
    Type t;
    t.kind = Kind::Unsupported;
    t.name = std::move(cppName);
    return t;
}

Type::Kind Type::scalarKind() const {
    switch (kind) {
        case Kind::Bool: case Kind::Int: case Kind::UInt: case Kind::Float: return kind;
        case Kind::Float2: case Kind::Float3: case Kind::Float4: return Kind::Float;
        case Kind::Int2: case Kind::Int3: case Kind::Int4: return Kind::Int;
        case Kind::UInt2: case Kind::UInt3: case Kind::UInt4: return Kind::UInt;
        default: return Kind::Unknown;
    }
}

unsigned Type::componentCount() const {
    if (isScalar()) return 1;
    switch (kind) {
        case Kind::Float2: case Kind::Int2: case Kind::UInt2: return 2;
        case Kind::Float3: case Kind::Int3: case Kind::UInt3: return 3;
        case Kind::Float4: case Kind::Int4: case Kind::UInt4: return 4;
        default: return 0;
    }
}

bool Type::sameAs(const Type& o) const {
    if (kind != o.kind) return false;
    switch (kind) {
        case Kind::Pointer:
            return inner && o.inner && inner->sameAs(*o.inner);
        case Kind::Array:
            return arraySize == o.arraySize && inner && o.inner && inner->sameAs(*o.inner);
        case Kind::Struct:
        case Kind::Unsupported:
            return name == o.name;
        default:
            return true;
    }
}

std::string Type::toShaderString() const {
    switch (kind) {
        case Kind::Void: return "void";
        case Kind::Bool: return "bool";
        case Kind::Int: return "int";
        case Kind::UInt: return "uint";
        case Kind::Float: return "float";
        case Kind::Float2: return "vec2";
        case Kind::Float3: return "vec3";
        case Kind::Float4: return "vec4";
        case Kind::Int2: return "ivec2";
        case Kind::Int3: return "ivec3";
        case Kind::Int4: return "ivec4";
        case Kind::UInt2: return "uvec2";
        case Kind::UInt3: return "uvec3";
        case Kind::UInt4: return "uvec4";
        case Kind::Struct: return name;
        case Kind::Array: return inner ? inner->toShaderString() : "float";
        case Kind::Pointer: return inner ? inner->toShaderString() : "float";
        default: return "float";
    }
}

std::string Type::toCppString() const {
    std::string s;
    switch (kind) {
        case Kind::Void: s = "void"; break;
        case Kind::Bool: s = "bool"; break;
        case Kind::Int: s = "int"; break;
        case Kind::UInt: s = "unsigned int"; break;
        case Kind::Float: s = "float"; break;
        case Kind::Float2: s = "float2"; break;
        case Kind::Float3: s = "float3"; break;
        case Kind::Float4: s = "float4"; break;
        case Kind::Int2: s = "int2"; break;
        case Kind::Int3: s = "int3"; break;
        case Kind::Int4: s = "int4"; break;
        case Kind::UInt2: s = "uint2"; break;
        case Kind::UInt3: s = "uint3"; break;
        case Kind::UInt4: s = "uint4"; break;
        case Kind::Struct: s = name; break;
        case Kind::Unsupported: s = name; break;
        case Kind::Pointer: {
            Type elem = inner ? *inner : Type::makeVoid();
            elem.isConst = false; // the constness of the pointee is carried by the pointer itself
            return (isConst ? "const " : "") + elem.toCppString() + "*";
        }
        case Kind::Array:
            return (inner ? inner->toCppString() : std::string("void")) + "[" + std::to_string(arraySize) + "]";
        case Kind::Unknown: s = "<unknown>"; break;
    }
    return (isConst && !isPointer() ? "const " : "") + s;
}

unsigned Type::sizeInBytes() const {
    switch (kind) {
        case Kind::Bool: case Kind::Int: case Kind::UInt: case Kind::Float: return 4;
        case Kind::Float2: case Kind::Int2: case Kind::UInt2: return 8;
        case Kind::Float3: case Kind::Int3: case Kind::UInt3:
        case Kind::Float4: case Kind::Int4: case Kind::UInt4: return 16;
        case Kind::Array: return inner ? arraySize * inner->sizeInBytes() : 0;
        default: return 0;
    }
}

const char* binaryOpText(BinaryOp op) {
    switch (op) {
        case BinaryOp::Add: return "+";
        case BinaryOp::Sub: return "-";
        case BinaryOp::Mul: return "*";
        case BinaryOp::Div: return "/";
        case BinaryOp::Mod: return "%";
        case BinaryOp::Shl: return "<<";
        case BinaryOp::Shr: return ">>";
        case BinaryOp::BitAnd: return "&";
        case BinaryOp::BitOr: return "|";
        case BinaryOp::BitXor: return "^";
        case BinaryOp::LogAnd: return "&&";
        case BinaryOp::LogOr: return "||";
        case BinaryOp::Eq: return "==";
        case BinaryOp::Ne: return "!=";
        case BinaryOp::Lt: return "<";
        case BinaryOp::Le: return "<=";
        case BinaryOp::Gt: return ">";
        case BinaryOp::Ge: return ">=";
        case BinaryOp::Assign: return "=";
        case BinaryOp::AddAssign: return "+=";
        case BinaryOp::SubAssign: return "-=";
        case BinaryOp::MulAssign: return "*=";
        case BinaryOp::DivAssign: return "/=";
        case BinaryOp::ModAssign: return "%=";
        case BinaryOp::ShlAssign: return "<<=";
        case BinaryOp::ShrAssign: return ">>=";
        case BinaryOp::AndAssign: return "&=";
        case BinaryOp::OrAssign: return "|=";
        case BinaryOp::XorAssign: return "^=";
        case BinaryOp::Comma: return ",";
    }
    return "?";
}

bool isAssignmentOp(BinaryOp op) { return op >= BinaryOp::Assign && op <= BinaryOp::XorAssign; }
bool isCompoundAssignmentOp(BinaryOp op) { return op > BinaryOp::Assign && op <= BinaryOp::XorAssign; }
bool isComparisonOp(BinaryOp op) { return op >= BinaryOp::Eq && op <= BinaryOp::Ge; }
bool isLogicalOp(BinaryOp op) { return op == BinaryOp::LogAnd || op == BinaryOp::LogOr; }

BinaryOp compoundBase(BinaryOp op) {
    switch (op) {
        case BinaryOp::AddAssign: return BinaryOp::Add;
        case BinaryOp::SubAssign: return BinaryOp::Sub;
        case BinaryOp::MulAssign: return BinaryOp::Mul;
        case BinaryOp::DivAssign: return BinaryOp::Div;
        case BinaryOp::ModAssign: return BinaryOp::Mod;
        case BinaryOp::ShlAssign: return BinaryOp::Shl;
        case BinaryOp::ShrAssign: return BinaryOp::Shr;
        case BinaryOp::AndAssign: return BinaryOp::BitAnd;
        case BinaryOp::OrAssign: return BinaryOp::BitOr;
        case BinaryOp::XorAssign: return BinaryOp::BitXor;
        default: return op;
    }
}

const char* unaryOpText(UnaryOp op) {
    switch (op) {
        case UnaryOp::Plus: return "+";
        case UnaryOp::Minus: return "-";
        case UnaryOp::LogNot: return "!";
        case UnaryOp::BitNot: return "~";
        case UnaryOp::PreInc: case UnaryOp::PostInc: return "++";
        case UnaryOp::PreDec: case UnaryOp::PostDec: return "--";
        case UnaryOp::Deref: return "*";
        case UnaryOp::AddressOf: return "&";
    }
    return "?";
}

const FunctionDecl* ParsedModule::findFunction(const std::string& name) const {
    const FunctionDecl* proto = nullptr;
    for (const auto& f : functions) {
        if (f->name != name) continue;
        if (f->body) return f.get();
        proto = f.get();
    }
    return proto;
}

} // namespace cudagfx

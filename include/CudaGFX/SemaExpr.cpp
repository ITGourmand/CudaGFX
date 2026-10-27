// SemaExpr.cpp: typing of expressions, explicit conversions, atomics.
#include "Sema.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include "Intrinsics.hpp"
#include "ShaderNames.hpp"

namespace cudagfx {

namespace {

int scalarRank(Type::Kind k) {
    switch (k) {
        case Type::Kind::Float: return 3;
        case Type::Kind::UInt: return 2;
        default: return 1; // Bool and Int: integer promotion
    }
}

// Common scalar type (C rules: bool -> int, int & uint -> uint, anything & float -> float).
Type commonScalar(const Type& a, const Type& b) {
    const int r = std::max(scalarRank(a.kind), scalarRank(b.kind));
    return r == 3 ? Type::makeFloat() : (r == 2 ? Type::makeUInt() : Type::makeInt());
}

bool isIntegerLike(const Type& t) { return t.kind == Type::Kind::Int || t.kind == Type::Kind::UInt || t.kind == Type::Kind::Bool; }

const char* useName(bool write) { return write ? "write" : "read"; }

} // namespace

// ---------------------------------------------------------------------------------------------
// Conversion utilities
// ---------------------------------------------------------------------------------------------

ExprPtr Sema::makeErrorExpr(const SourceLocation& loc) const {
    auto e = std::make_unique<LiteralExpr>();
    e->loc = loc;
    e->lit = LiteralExpr::Lit::Int;
    e->type = Type::makeUnknown();
    return e;
}

void Sema::wrapCast(ExprPtr& e, const Type& target, bool implicit) {
    if (e->kind == ExprKind::Literal) {
        auto& lit = static_cast<LiteralExpr&>(*e);
        const bool isIntLit = lit.lit == LiteralExpr::Lit::Int || lit.lit == LiteralExpr::Lit::UInt;
        if (target.kind == Type::Kind::Float && isIntLit) {
            lit.floatValue = lit.lit == LiteralExpr::Lit::UInt ? static_cast<double>(static_cast<std::uint32_t>(lit.intValue))
                                                              : static_cast<double>(lit.intValue);
            lit.lit = LiteralExpr::Lit::Float;
            lit.type = target;
            return;
        }
        if (target.kind == Type::Kind::UInt && lit.lit == LiteralExpr::Lit::Int && lit.intValue >= 0) {
            lit.lit = LiteralExpr::Lit::UInt;
            lit.type = target;
            return;
        }
        if (target.kind == Type::Kind::Int && lit.lit == LiteralExpr::Lit::UInt && lit.intValue <= INT_MAX) {
            lit.lit = LiteralExpr::Lit::Int;
            lit.type = target;
            return;
        }
    }
    auto c = std::make_unique<CastExpr>();
    c->loc = e->loc;
    c->type = target;
    c->implicit = implicit;
    c->operand = std::move(e);
    e = std::move(c);
}

bool Sema::coerce(ExprPtr& e, const Type& target, const SourceLocation& loc, const char* what) {
    const Type et = e->type;
    if (et.isUnknown() || target.isUnknown()) return false;
    if (et.isVoid() || target.isVoid()) {
        diags_.reportError(DiagCode::CGFX0019, loc, std::string(what) + ": void value");
        return false;
    }
    if (et.sameAs(target)) return true;
    if (et.isScalar() && target.isScalar()) {
        wrapCast(e, target, true);
        return true;
    }
    diags_.reportError(DiagCode::CGFX0019, loc,
                       std::string(what) + ": cannot convert '" + et.toCppString() + "' to '" + target.toCppString() + "'");
    return false;
}

void Sema::toBool(ExprPtr& e) {
    const Type t = e->type;
    if (t.isUnknown() || t.kind == Type::Kind::Bool) return;
    if (t.isScalar()) {
        wrapCast(e, Type::makeBool(), true);
        return;
    }
    diags_.reportError(DiagCode::CGFX0019, e->loc, "Condition of type '" + t.toCppString() + "' is invalid (boolean or scalar expected)");
}

bool Sema::isLValue(const Expr& e) const {
    switch (e.kind) {
        case ExprKind::VarRef: {
            const auto& r = static_cast<const VarRefExpr&>(e);
            return r.symbol == SymbolKind::Local || r.symbol == SymbolKind::ScalarParam || r.symbol == SymbolKind::Shared;
        }
        case ExprKind::Index: return true;
        case ExprKind::Member: return isLValue(*static_cast<const MemberExpr&>(e).base);
        case ExprKind::Paren: return isLValue(*static_cast<const ParenExpr&>(e).sub);
        default: return false;
    }
}

bool Sema::isPureExpr(const Expr& e) const {
    switch (e.kind) {
        case ExprKind::Literal: case ExprKind::VarRef: case ExprKind::Builtin: return true;
        case ExprKind::Binary: {
            const auto& b = static_cast<const BinaryExpr&>(e);
            return !isAssignmentOp(b.op) && b.op != BinaryOp::Comma && isPureExpr(*b.left) && isPureExpr(*b.right);
        }
        case ExprKind::Unary: {
            const auto& u = static_cast<const UnaryExpr&>(e);
            return !isIncDecOp(u.op) && u.op != UnaryOp::Deref && u.op != UnaryOp::AddressOf && isPureExpr(*u.operand);
        }
        case ExprKind::Paren: return isPureExpr(*static_cast<const ParenExpr&>(e).sub);
        case ExprKind::Index: {
            const auto& i = static_cast<const IndexExpr&>(e);
            return isPureExpr(*i.base) && isPureExpr(*i.index);
        }
        case ExprKind::Member: return isPureExpr(*static_cast<const MemberExpr&>(e).base);
        case ExprKind::Cast: return isPureExpr(*static_cast<const CastExpr&>(e).operand);
        case ExprKind::Ternary: {
            const auto& t = static_cast<const TernaryExpr&>(e);
            return isPureExpr(*t.cond) && isPureExpr(*t.thenExpr) && isPureExpr(*t.elseExpr);
        }
        default: return false;
    }
}

ExprPtr Sema::cloneExpr(const Expr& e) const {
    ExprPtr out;
    switch (e.kind) {
        case ExprKind::Literal: out = std::make_unique<LiteralExpr>(static_cast<const LiteralExpr&>(e)); break;
        case ExprKind::VarRef: out = std::make_unique<VarRefExpr>(static_cast<const VarRefExpr&>(e)); break;
        case ExprKind::Builtin: out = std::make_unique<BuiltinExpr>(static_cast<const BuiltinExpr&>(e)); break;
        case ExprKind::Binary: {
            const auto& s = static_cast<const BinaryExpr&>(e);
            auto n = std::make_unique<BinaryExpr>();
            n->op = s.op;
            n->left = cloneExpr(*s.left);
            n->right = cloneExpr(*s.right);
            out = std::move(n);
            break;
        }
        case ExprKind::Unary: {
            const auto& s = static_cast<const UnaryExpr&>(e);
            auto n = std::make_unique<UnaryExpr>();
            n->op = s.op;
            n->operand = cloneExpr(*s.operand);
            out = std::move(n);
            break;
        }
        case ExprKind::Paren: {
            auto n = std::make_unique<ParenExpr>();
            n->sub = cloneExpr(*static_cast<const ParenExpr&>(e).sub);
            out = std::move(n);
            break;
        }
        case ExprKind::Index: {
            const auto& s = static_cast<const IndexExpr&>(e);
            auto n = std::make_unique<IndexExpr>();
            n->base = cloneExpr(*s.base);
            n->index = cloneExpr(*s.index);
            out = std::move(n);
            break;
        }
        case ExprKind::Member: {
            const auto& s = static_cast<const MemberExpr&>(e);
            auto n = std::make_unique<MemberExpr>();
            n->base = cloneExpr(*s.base);
            n->member = s.member;
            n->viaArrow = s.viaArrow;
            n->emitMember = s.emitMember;
            out = std::move(n);
            break;
        }
        case ExprKind::Cast: {
            const auto& s = static_cast<const CastExpr&>(e);
            auto n = std::make_unique<CastExpr>();
            n->operand = cloneExpr(*s.operand);
            n->implicit = s.implicit;
            out = std::move(n);
            break;
        }
        case ExprKind::Ternary: {
            const auto& s = static_cast<const TernaryExpr&>(e);
            auto n = std::make_unique<TernaryExpr>();
            n->cond = cloneExpr(*s.cond);
            n->thenExpr = cloneExpr(*s.thenExpr);
            n->elseExpr = cloneExpr(*s.elseExpr);
            out = std::move(n);
            break;
        }
        default:
            return makeErrorExpr(e.loc);
    }
    out->loc = e.loc;
    out->type = e.type;
    return out;
}

// ---------------------------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------------------------

void Sema::analyzeExpr(ExprPtr& e, Use use) {
    switch (e->kind) {
        case ExprKind::Literal: analyzeLiteral(static_cast<LiteralExpr&>(*e)); break;
        case ExprKind::VarRef: analyzeVarRef(static_cast<VarRefExpr&>(*e), use, false); break;
        case ExprKind::Builtin: analyzeBuiltin(static_cast<BuiltinExpr&>(*e), use); break;
        case ExprKind::Binary: analyzeBinary(e); break;
        case ExprKind::Unary: analyzeUnary(e); break;
        case ExprKind::Paren: {
            auto& p = static_cast<ParenExpr&>(*e);
            analyzeExpr(p.sub, use);
            p.type = p.sub->type;
            break;
        }
        case ExprKind::Index: analyzeIndex(static_cast<IndexExpr&>(*e), use); break;
        case ExprKind::Member: analyzeMember(static_cast<MemberExpr&>(*e), use); break;
        case ExprKind::Call: analyzeCall(static_cast<CallExpr&>(*e)); break;
        case ExprKind::Cast: analyzeCast(static_cast<CastExpr&>(*e)); break;
        case ExprKind::Ternary: analyzeTernary(static_cast<TernaryExpr&>(*e)); break;
        case ExprKind::Atomic: break; // already validated by buildAtomic
    }
}

// ---------------------------------------------------------------------------------------------
// Leaves
// ---------------------------------------------------------------------------------------------

void Sema::analyzeLiteral(LiteralExpr& lit) {
    switch (lit.lit) {
        case LiteralExpr::Lit::Int: if (!lit.type.isUnknown()) lit.type = Type::makeInt(); break;
        case LiteralExpr::Lit::UInt: lit.type = Type::makeUInt(); break;
        case LiteralExpr::Lit::Float: lit.type = Type::makeFloat(); break;
        case LiteralExpr::Lit::Bool: lit.type = Type::makeBool(); break;
        case LiteralExpr::Lit::Unsupported: {
            const std::string& t = lit.text;
            if (!t.empty() && (t[0] == '"' || t.find('"') != std::string::npos)) {
                diags_.reportError(DiagCode::CGFX0009, lit.loc, "String literals are not supported in device code");
            } else if (!t.empty() && t[0] == '\'') {
                diags_.reportError(DiagCode::CGFX0019, lit.loc, "Character literal not supported: " + t);
            } else {
                diags_.reportError(DiagCode::CGFX0002, lit.loc,
                                   "Literal outside the 32-bit subset ('" + t + "'): L/LL suffixes, values > 32 bits and long double are not supported");
            }
            lit.type = Type::makeUnknown();
            break;
        }
    }
}

void Sema::analyzeVarRef(VarRefExpr& ref, Use use, bool pointerOk) {
    Symbol globalSym;
    const Symbol* s = lookup(ref.name);
    if (!s && lookupGlobalConst(ref.name, ref.loc, globalSym)) s = &globalSym;
    if (!s) {
        ref.type = Type::makeUnknown();
        // lookupGlobalConst has already explained the failure of an invalid constant.
        const auto g = globals_.find(ref.name);
        if (g != globals_.end() && g->second.decl) return;
        if (ref.name == "nullptr" || ref.name == "NULL") {
            diags_.reportError(DiagCode::CGFX0001, ref.loc, "Null pointer is not supported in a kernel");
        } else if (isWarpIntrinsic(ref.name)) {
            diags_.reportError(DiagCode::CGFX0003, ref.loc, "'" + ref.name + "' (warp) is not supported");
        } else if (functions_.count(ref.name)) {
            diags_.reportError(DiagCode::CGFX0019, ref.loc, "Function '" + ref.name + "' is used as a value");
        } else {
            diags_.reportError(DiagCode::CGFX0010, ref.loc, "Undeclared identifier: '" + ref.name + "'");
        }
        return;
    }
    ref.symbol = s->kind;
    ref.type = s->type;
    ref.emitName = s->emitName;
    ref.isConstData = s->isConst;

    if (s->kind == SymbolKind::BufferParam) {
        if (!pointerOk) {
            diags_.reportError(DiagCode::CGFX0001, ref.loc,
                               "Pointer '" + ref.name + "' is used as a value (arithmetic, alias, assignment, argument passing): "
                               "only indexing p[i] is supported");
            ref.type = Type::makeUnknown();
            return;
        }
        if (s->param) {
            if (use != Use::Read) {
                if (s->isConst) {
                    diags_.reportError(DiagCode::CGFX0019, ref.loc, "Write to const buffer '" + ref.name + "'");
                } else {
                    s->param->isWritten = true;
                }
            }
            if (use != Use::Write) s->param->isRead = true;
        }
        return;
    }
    if (use != Use::Read && s->isConst) {
        diags_.reportError(DiagCode::CGFX0019, ref.loc,
                           std::string("Assignment to const variable '") + ref.name + "' (" + useName(true) + ")");
    }
}

void Sema::analyzeBuiltin(BuiltinExpr& b, Use use) {
    if (use != Use::Read) {
        diags_.reportError(DiagCode::CGFX0019, b.loc, "The built-in variables (threadIdx, blockIdx, blockDim, gridDim) are read-only");
    }
    b.type = b.component ? Type::makeUInt() : Type::makeVector(Type::Kind::UInt, 3);
    if (b.builtin == BuiltinExpr::Kind::GridDim) usesGridDim_ = true;
    if (inGlobalInit_) {
        diags_.reportError(DiagCode::CGFX0019, b.loc, "Built-in variable not allowed in a global initializer");
    }
}

// ---------------------------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------------------------

Sema::BinaryPlan Sema::planBinary(BinaryOp op, const Type& lt, const Type& rt) const {
    BinaryPlan plan;
    plan.lhsTarget = Type::makeUnknown();
    plan.rhsTarget = Type::makeUnknown();
    auto fail = [&](const char* why) {
        plan.ok = false;
        plan.why = why;
        return plan;
    };
    const bool lNum = lt.isScalar();
    const bool rNum = rt.isScalar();

    if (isLogicalOp(op)) {
        if (!lNum || !rNum) return fail("scalar operands expected");
        plan.ok = true;
        plan.result = Type::makeBool();
        plan.lhsTarget = plan.rhsTarget = Type::makeBool();
        return plan;
    }
    if (isComparisonOp(op)) {
        if (!lNum || !rNum) return fail("comparison of scalars only (compare the components of a vector one by one)");
        plan.ok = true;
        plan.result = Type::makeBool();
        if ((op == BinaryOp::Eq || op == BinaryOp::Ne) && lt.kind == Type::Kind::Bool && rt.kind == Type::Kind::Bool) return plan;
        plan.lhsTarget = plan.rhsTarget = commonScalar(lt, rt);
        return plan;
    }
    if (op == BinaryOp::Shl || op == BinaryOp::Shr) {
        if (!lNum || !rNum || lt.kind == Type::Kind::Float || rt.kind == Type::Kind::Float) return fail("shift: integers expected");
        plan.ok = true;
        plan.result = lt.kind == Type::Kind::Bool ? Type::makeInt() : lt;
        plan.lhsTarget = plan.rhsTarget = plan.result;
        return plan;
    }
    if (op == BinaryOp::Mod || op == BinaryOp::BitAnd || op == BinaryOp::BitOr || op == BinaryOp::BitXor) {
        if (!lNum || !rNum || !isIntegerLike(lt) || !isIntegerLike(rt)) {
            return fail(op == BinaryOp::Mod ? "'%' only exists on integers (use fmodf for floating-point values)"
                                            : "bitwise operator: integers expected");
        }
        plan.ok = true;
        plan.result = commonScalar(lt, rt);
        plan.lhsTarget = plan.rhsTarget = plan.result;
        return plan;
    }
    // + - * /
    if (lNum && rNum) {
        plan.ok = true;
        plan.result = commonScalar(lt, rt);
        plan.lhsTarget = plan.rhsTarget = plan.result;
        return plan;
    }
    if (lt.isVector() && rt.isVector()) {
        if (!lt.sameAs(rt)) return fail("vectors of different types");
        plan.ok = true;
        plan.result = lt;
        return plan;
    }
    if (lt.isVector() && rNum) {
        const Type comp = Type::scalar(lt.scalarKind());
        if (rt.kind == Type::Kind::Float && comp.kind != Type::Kind::Float) return fail("floating-point scalar with an integer vector");
        plan.ok = true;
        plan.result = lt;
        plan.rhsTarget = comp;
        return plan;
    }
    if (lNum && rt.isVector()) {
        const Type comp = Type::scalar(rt.scalarKind());
        if (lt.kind == Type::Kind::Float && comp.kind != Type::Kind::Float) return fail("floating-point scalar with an integer vector");
        plan.ok = true;
        plan.result = rt;
        plan.lhsTarget = comp;
        return plan;
    }
    return fail("non-arithmetic types");
}

Type Sema::typeBinaryOp(BinaryOp op, ExprPtr& l, ExprPtr& r, const SourceLocation& loc) {
    if (l->type.isUnknown() || r->type.isUnknown()) return Type::makeUnknown();
    const Type lt = l->type;
    const Type rt = r->type;
    const BinaryPlan p = planBinary(op, lt, rt);
    if (!p.ok) {
        diags_.reportError(DiagCode::CGFX0019, loc,
                           std::string("Invalid operands for '") + binaryOpText(op) + "': " + p.why + " (" +
                               lt.toCppString() + " and " + rt.toCppString() + ")");
        return Type::makeUnknown();
    }
    if (!p.lhsTarget.isUnknown() && !l->type.sameAs(p.lhsTarget)) wrapCast(l, p.lhsTarget, true);
    if (!p.rhsTarget.isUnknown() && !r->type.sameAs(p.rhsTarget)) wrapCast(r, p.rhsTarget, true);
    if ((op == BinaryOp::Div || op == BinaryOp::Mod) && p.result.kind == Type::Kind::Int &&
        (lt.kind == Type::Kind::Int || lt.kind == Type::Kind::Bool) && (rt.kind == Type::Kind::Int || rt.kind == Type::Kind::Bool)) {
        diags_.reportWarning(DiagCode::CGFX0017, loc,
                             std::string("'") + binaryOpText(op) + "' on signed integers: the result differs between backends for negative operands");
    }
    return p.result;
}

void Sema::analyzeBinary(ExprPtr& ep) {
    auto& b = static_cast<BinaryExpr&>(*ep);
    if (b.op == BinaryOp::Comma) {
        analyzeExpr(b.left);
        analyzeExpr(b.right);
        b.type = b.right->type;
        return;
    }
    if (isAssignmentOp(b.op)) {
        analyzeAssignment(ep);
        return;
    }
    analyzeExpr(b.left);
    analyzeExpr(b.right);
    b.type = typeBinaryOp(b.op, b.left, b.right, b.loc);
}

void Sema::analyzeAssignment(ExprPtr& ep) {
    auto& b = static_cast<BinaryExpr&>(*ep);
    const bool compound = isCompoundAssignmentOp(b.op);
    analyzeExpr(b.left, compound ? Use::ReadWrite : Use::Write);
    analyzeExpr(b.right, Use::Read);
    const Type lt = b.left->type;
    b.type = lt;
    if (lt.isUnknown() || b.right->type.isUnknown()) return;
    if (!isLValue(*b.left)) {
        diags_.reportError(DiagCode::CGFX0019, b.loc, "The left-hand side of the assignment is not assignable");
        b.type = Type::makeUnknown();
        return;
    }
    if (lt.isArray() || lt.isPointer()) {
        diags_.reportError(DiagCode::CGFX0019, b.loc, "Assignment to an array or a pointer is not supported");
        b.type = Type::makeUnknown();
        return;
    }
    if (!compound) {
        coerce(b.right, lt, b.loc, "assignment");
        return;
    }

    const BinaryOp base = compoundBase(b.op);
    const BinaryPlan plan = planBinary(base, lt, b.right->type);
    if (!plan.ok) {
        diags_.reportError(DiagCode::CGFX0019, b.loc,
                           std::string("Invalid operands for '") + binaryOpText(b.op) + "': " + plan.why + " (" +
                               lt.toCppString() + " and " + b.right->type.toCppString() + ")");
        b.type = Type::makeUnknown();
        return;
    }
    if (plan.result.sameAs(lt)) {
        if (!plan.rhsTarget.isUnknown() && !b.right->type.sameAs(plan.rhsTarget)) wrapCast(b.right, plan.rhsTarget, true);
        if ((base == BinaryOp::Div || base == BinaryOp::Mod) && lt.kind == Type::Kind::Int && b.right->type.kind == Type::Kind::Int) {
            diags_.reportWarning(DiagCode::CGFX0017, b.loc,
                                 std::string("'") + binaryOpText(b.op) + "' on signed integers: the result differs between backends for negative operands");
        }
        return;
    }

    // Implicit conversion with widening (int += float, bool |= int...): x = (T)(x op y), as in C.
    if (!isPureExpr(*b.left)) {
        diags_.reportError(DiagCode::CGFX0019, b.loc,
                           std::string("Compound assignment '") + binaryOpText(b.op) + "' with an implicit conversion on an expression with side effects: write x = (T)(x op y)");
        b.type = Type::makeUnknown();
        return;
    }
    auto bin = std::make_unique<BinaryExpr>();
    bin->loc = b.loc;
    bin->op = base;
    bin->left = cloneExpr(*b.left);
    bin->right = std::move(b.right);
    bin->type = typeBinaryOp(base, bin->left, bin->right, b.loc);
    ExprPtr value = std::move(bin);
    if (!value->type.isUnknown() && !value->type.sameAs(lt)) wrapCast(value, lt, true);
    b.op = BinaryOp::Assign;
    b.right = std::move(value);
}

void Sema::analyzeUnary(ExprPtr& ep) {
    auto& u = static_cast<UnaryExpr&>(*ep);
    switch (u.op) {
        case UnaryOp::Deref:
            diags_.reportError(DiagCode::CGFX0001, u.loc, "Dereference '*p' is not supported: use indexing p[i]");
            u.type = Type::makeUnknown();
            return;
        case UnaryOp::AddressOf:
            diags_.reportError(DiagCode::CGFX0001, u.loc,
                               "Address-of operator '&' is not supported (except as the first argument of atomics: atomicAdd(&a[i], v))");
            u.type = Type::makeUnknown();
            return;
        case UnaryOp::LogNot:
            analyzeExpr(u.operand);
            toBool(u.operand);
            u.type = u.operand->type.isUnknown() ? Type::makeUnknown() : Type::makeBool();
            return;
        case UnaryOp::PreInc: case UnaryOp::PreDec: case UnaryOp::PostInc: case UnaryOp::PostDec: {
            analyzeExpr(u.operand, Use::ReadWrite);
            const Type t = u.operand->type;
            u.type = t;
            if (t.isUnknown()) return;
            if (!isLValue(*u.operand)) {
                diags_.reportError(DiagCode::CGFX0019, u.loc, "The operand of ++/-- is not assignable");
                u.type = Type::makeUnknown();
            } else if (!(t.kind == Type::Kind::Int || t.kind == Type::Kind::UInt || t.kind == Type::Kind::Float)) {
                diags_.reportError(DiagCode::CGFX0019, u.loc, "++/-- on a non-numeric type ('" + t.toCppString() + "')");
                u.type = Type::makeUnknown();
            }
            return;
        }
        case UnaryOp::BitNot: {
            analyzeExpr(u.operand);
            const Type t = u.operand->type;
            if (t.isUnknown()) { u.type = t; return; }
            if (!t.isScalar() || !isIntegerLike(t)) {
                diags_.reportError(DiagCode::CGFX0019, u.loc, "'~' on a non-integer type ('" + t.toCppString() + "')");
                u.type = Type::makeUnknown();
                return;
            }
            if (t.kind == Type::Kind::Bool) wrapCast(u.operand, Type::makeInt(), true);
            u.type = u.operand->type;
            return;
        }
        case UnaryOp::Plus:
        case UnaryOp::Minus: {
            analyzeExpr(u.operand);
            const Type t = u.operand->type;
            if (t.isUnknown()) { u.type = t; return; }
            if (t.kind == Type::Kind::Bool) wrapCast(u.operand, Type::makeInt(), true);
            else if (!t.isScalar() && !t.isVector()) {
                diags_.reportError(DiagCode::CGFX0019, u.loc, std::string("'") + unaryOpText(u.op) + "' on a non-numeric type ('" + t.toCppString() + "')");
                u.type = Type::makeUnknown();
                return;
            }
            u.type = u.operand->type;
            return;
        }
    }
}

void Sema::analyzeIndex(IndexExpr& ix, Use use) {
    if (ix.base->kind == ExprKind::VarRef) analyzeVarRef(static_cast<VarRefExpr&>(*ix.base), use, true);
    else analyzeExpr(ix.base, use);
    analyzeExpr(ix.index, Use::Read);

    const Type bt = ix.base->type;
    const Type it = ix.index->type;
    if (!it.isUnknown()) {
        if (!it.isScalar() || it.kind == Type::Kind::Float) {
            diags_.reportError(DiagCode::CGFX0019, ix.index->loc, "The index must be an integer (type '" + it.toCppString() + "')");
        } else if (it.kind == Type::Kind::Bool) {
            wrapCast(ix.index, Type::makeInt(), true);
        }
    }
    if (bt.isUnknown()) {
        ix.type = Type::makeUnknown();
        return;
    }
    if ((bt.isPointer() || bt.isArray()) && bt.inner) {
        ix.type = *bt.inner;
        if (bt.isConst && !ix.type.isConst) ix.type.isConst = bt.isConst;
    } else if (bt.isVector()) {
        ix.type = Type::scalar(bt.scalarKind());
    } else {
        diags_.reportError(DiagCode::CGFX0019, ix.loc, "Indexing a non-indexable type ('" + bt.toCppString() + "')");
        ix.type = Type::makeUnknown();
    }
}

void Sema::analyzeMember(MemberExpr& m, Use use) {
    analyzeExpr(m.base, use);
    const Type bt = m.base->type;
    if (bt.isUnknown()) {
        m.type = Type::makeUnknown();
        return;
    }
    if (m.viaArrow || bt.isPointer()) {
        diags_.reportError(DiagCode::CGFX0001, m.loc, "Pointer member access ('->') is not supported: pointers are only used to index buffers");
        m.type = Type::makeUnknown();
        return;
    }
    if (bt.isVector()) {
        const unsigned comps = bt.componentCount();
        const std::string& name = m.member;
        bool ok = !name.empty() && name.size() <= 4;
        for (char c : name) {
            const std::size_t idx = std::string("xyzw").find(c);
            if (idx == std::string::npos || idx >= comps) ok = false;
        }
        if (!ok) {
            diags_.reportError(DiagCode::CGFX0019, m.loc,
                               "Component '." + name + "' is invalid for type '" + bt.toCppString() + "'");
            m.type = Type::makeUnknown();
            return;
        }
        m.type = name.size() == 1 ? Type::scalar(bt.scalarKind()) : Type::makeVector(bt.scalarKind(), static_cast<unsigned>(name.size()));
        m.emitMember = name;
        if (bt.isConst) m.type.isConst = true;
        return;
    }
    if (bt.isStruct()) {
        const StructDecl* sd = findStruct(bt.name);
        if (sd) {
            for (const auto& f : sd->fields) {
                if (f.name == m.member) {
                    m.type = f.type;
                    m.emitMember = f.emitName;
                    return;
                }
            }
        }
        diags_.reportError(DiagCode::CGFX0019, m.loc, "Struct '" + bt.name + "' has no member '" + m.member + "'");
        m.type = Type::makeUnknown();
        return;
    }
    diags_.reportError(DiagCode::CGFX0019, m.loc, "Member access on a type without members ('" + bt.toCppString() + "')");
    m.type = Type::makeUnknown();
}

void Sema::analyzeCast(CastExpr& c) {
    c.type = resolveType(c.type, c.loc);
    analyzeExpr(c.operand);
    const Type from = c.operand->type;
    const Type to = c.type;
    if (to.isUnknown() || from.isUnknown()) {
        c.type = to.isUnknown() ? Type::makeUnknown() : to;
        return;
    }
    if (to.isVoid()) return;
    if (to.isPointer()) {
        diags_.reportError(DiagCode::CGFX0001, c.loc, "Conversion to a pointer is not supported");
        c.type = Type::makeUnknown();
    } else if (to.isScalar() && from.isScalar()) {
        return;
    } else if (to.isVector() && from.sameAs(to)) {
        return;
    } else {
        diags_.reportError(DiagCode::CGFX0019, c.loc,
                           "Unsupported conversion from '" + from.toCppString() + "' to '" + to.toCppString() + "'");
        c.type = Type::makeUnknown();
    }
}

void Sema::analyzeTernary(TernaryExpr& t) {
    analyzeExpr(t.cond);
    toBool(t.cond);
    analyzeExpr(t.thenExpr);
    analyzeExpr(t.elseExpr);
    const Type a = t.thenExpr->type;
    const Type b = t.elseExpr->type;
    if (a.isUnknown() || b.isUnknown()) {
        t.type = Type::makeUnknown();
        return;
    }
    if (a.isScalar() && b.isScalar()) {
        Type common = (a.kind == Type::Kind::Bool && b.kind == Type::Kind::Bool) ? a : commonScalar(a, b);
        if (!a.sameAs(common)) wrapCast(t.thenExpr, common, true);
        if (!b.sameAs(common)) wrapCast(t.elseExpr, common, true);
        t.type = common;
    } else if (a.sameAs(b) && (a.isVector() || a.isStruct())) {
        t.type = a;
    } else {
        diags_.reportError(DiagCode::CGFX0019, t.loc,
                           "Branches of '?:' have incompatible types ('" + a.toCppString() + "' and '" + b.toCppString() + "')");
        t.type = Type::makeUnknown();
    }
}

// ---------------------------------------------------------------------------------------------
// Calls
// ---------------------------------------------------------------------------------------------

void Sema::analyzeCall(CallExpr& c) {
    AtomicOp aop;
    if (parseAtomicName(c.callee, aop)) {
        diags_.reportError(DiagCode::CGFX0015, c.loc,
                           "The result of '" + c.callee + "' is used in the middle of an expression: "
                           "write 'int old = " + c.callee + "(&a[i], v);' or ignore the result");
        c.type = Type::makeUnknown();
        return;
    }
    const auto uf = functions_.find(c.callee);
    if (uf != functions_.end()) {
        analyzeUserCall(c, *uf->second);
        return;
    }
    if (const Intrinsic* in = findIntrinsic(c.callee)) {
        analyzeIntrinsicCall(c, *in);
        return;
    }
    // Arguments are not analyzed: the call is already invalid, no point piling up cascading errors.
    c.type = Type::makeUnknown();
    if (isWarpIntrinsic(c.callee)) {
        diags_.reportError(DiagCode::CGFX0003, c.loc, "Warp intrinsic '" + c.callee + "' is not supported");
    } else if (isUnsupportedDeviceFunction(c.callee)) {
        diags_.reportError(DiagCode::CGFX0005, c.loc, "Function '" + c.callee + "' is not supported in device code");
    } else {
        std::string hint;
        if ((c.callee.rfind("float", 0) == 0 || c.callee.rfind("int", 0) == 0 || c.callee.rfind("uint", 0) == 0) &&
            !c.callee.empty() && std::isdigit(static_cast<unsigned char>(c.callee.back()))) {
            hint = " (use make_" + c.callee + ")";
        } else if (c.callee.rfind("std::", 0) == 0) {
            hint = " (the std namespace is not available: use sqrtf, fminf, fabsf...)";
        }
        diags_.reportError(DiagCode::CGFX0010, c.loc,
                           "Unknown function: '" + c.callee + "' (neither a __device__ function defined in this file nor a supported native function)" + hint);
    }
}

void Sema::analyzeUserCall(CallExpr& c, const FunctionDecl& callee) {
    for (auto& a : c.args) analyzeExpr(a);
    c.target = &callee;
    c.emitName = safeIdentifier(callee.name);
    c.type = callee.returnType;
    if (inGlobalInit_) {
        diags_.reportError(DiagCode::CGFX0019, c.loc, "Function call not allowed in a global initializer");
        c.type = Type::makeUnknown();
        return;
    }
    if (callee.isKernel) {
        diags_.reportError(DiagCode::CGFX0009, c.loc,
                           "Direct call of kernel '" + callee.name + "' from device code: dynamic parallelism is not supported");
        c.type = Type::makeUnknown();
        return;
    }
    if (!callee.body) {
        diags_.reportError(DiagCode::CGFX0010, c.loc,
                           "__device__ function '" + callee.name + "' is declared but has no visible definition");
        c.type = Type::makeUnknown();
        return;
    }
    if (c.args.size() != callee.parameters.size()) {
        diags_.reportError(DiagCode::CGFX0019, c.loc,
                           "Function '" + callee.name + "' expects " + std::to_string(callee.parameters.size()) +
                               " argument(s), " + std::to_string(c.args.size()) + " provided");
        c.type = callee.returnType;
        return;
    }
    for (std::size_t i = 0; i < c.args.size(); ++i) {
        const Type& pt = callee.parameters[i].type;
        if (pt.isUnknown()) continue;
        coerce(c.args[i], pt, c.args[i]->loc, "argument");
    }
    if (current_) {
        callSites_[current_].push_back({&callee, c.loc});
        auto& v = current_->callees;
        if (std::find(v.begin(), v.end(), &callee) == v.end()) v.push_back(&callee);
    }
}

void Sema::analyzeIntrinsicCall(CallExpr& c, const Intrinsic& in) {
    for (auto& a : c.args) analyzeExpr(a);
    c.intrinsic = &in;
    c.emitName = in.shaderName;
    if (in.shaderName == kPopcountShaderName) usesPopcount_ = true;
    c.type = Type::makeUnknown();
    if (c.args.size() != in.argc) {
        diags_.reportError(DiagCode::CGFX0019, c.loc,
                           "'" + c.callee + "' expects " + std::to_string(in.argc) + " argument(s), " + std::to_string(c.args.size()) + " provided");
        return;
    }
    for (const auto& a : c.args) {
        if (a->type.isUnknown()) return;
        if (!a->type.isScalar()) {
            diags_.reportError(DiagCode::CGFX0019, a->loc, "Argument of '" + c.callee + "': scalar expected (type '" + a->type.toCppString() + "')");
            return;
        }
    }
    using Shape = Intrinsic::Shape;
    switch (in.shape) {
        case Shape::FloatUnary:
        case Shape::FloatBinary:
            for (auto& a : c.args) coerce(a, Type::makeFloat(), a->loc, "argument");
            c.type = Type::makeFloat();
            break;
        case Shape::Generic: {
            if (in.argc == 1) { // abs
                Type t = c.args[0]->type;
                if (t.kind == Type::Kind::UInt) {
                    diags_.reportError(DiagCode::CGFX0019, c.loc, "abs() on an unsigned integer makes no sense (ambiguous in C++)");
                    return;
                }
                if (t.kind == Type::Kind::Bool) wrapCast(c.args[0], Type::makeInt(), true);
                c.type = c.args[0]->type;
            } else {
                const Type common = commonScalar(c.args[0]->type, c.args[1]->type);
                for (auto& a : c.args) coerce(a, common, a->loc, "argument");
                c.type = common;
            }
            break;
        }
        case Shape::VectorMake: {
            const Type comp = Type::scalar(in.argKind);
            for (auto& a : c.args) coerce(a, comp, a->loc, "argument");
            c.type = Type::makeVector(in.retKind, in.components);
            c.emitName = c.type.toShaderString();
            break;
        }
        case Shape::Fixed:
            for (auto& a : c.args) coerce(a, Type::scalar(in.argKind), a->loc, "argument");
            c.type = Type::scalar(in.retKind);
            break;
    }
    if (in.warnSemantics) {
        diags_.reportWarning(DiagCode::CGFX0020, c.loc,
                             "'" + c.callee + "' is translated to mod(): the sign of the result differs from fmod for operands of opposite signs");
    }
}

ExprPtr Sema::buildAtomic(CallExpr& call, AtomicOp op, const std::string& resultName, const Type* resultType) {
    const SourceLocation loc = call.loc;
    const std::size_t need = op == AtomicOp::Cas ? 3 : 2;
    if (call.args.size() != need) {
        diags_.reportError(DiagCode::CGFX0015, loc, "'" + call.callee + "' expects " + std::to_string(need) + " arguments");
        return makeErrorExpr(loc);
    }
    Expr& a0 = *call.args[0];
    auto badTarget = [&]() {
        diags_.reportError(DiagCode::CGFX0015, a0.loc,
                           "The first argument of '" + call.callee +
                               "' must be the address of a buffer or shared-memory element: " + call.callee + "(&a[i], v)");
        return makeErrorExpr(loc);
    };
    if (a0.kind != ExprKind::Unary || static_cast<UnaryExpr&>(a0).op != UnaryOp::AddressOf) return badTarget();
    auto& addr = static_cast<UnaryExpr&>(a0);
    if (addr.operand->kind != ExprKind::Index) return badTarget();

    analyzeExpr(addr.operand, Use::ReadWrite);
    const Type elem = addr.operand->type;
    if (elem.isUnknown()) return makeErrorExpr(loc);

    const Expr* base = static_cast<IndexExpr&>(*addr.operand).base.get();
    while (base->kind == ExprKind::Paren) base = static_cast<const ParenExpr*>(base)->sub.get();
    const bool okBase = base->kind == ExprKind::VarRef &&
                        (static_cast<const VarRefExpr*>(base)->symbol == SymbolKind::BufferParam ||
                         static_cast<const VarRefExpr*>(base)->symbol == SymbolKind::Shared);
    if (!okBase) {
        diags_.reportError(DiagCode::CGFX0015, a0.loc, "Atomic on a local array is not allowed: only buffers and shared memory are atomic");
        return makeErrorExpr(loc);
    }
    if (!(elem.kind == Type::Kind::Int || elem.kind == Type::Kind::UInt)) {
        diags_.reportError(DiagCode::CGFX0015, a0.loc,
                           "Atomic on type '" + elem.toCppString() + "' is not supported (32-bit integers only)");
        return makeErrorExpr(loc);
    }

    auto at = std::make_unique<AtomicExpr>();
    at->loc = loc;
    at->op = op;
    at->type = elem;
    at->target = std::move(addr.operand);
    for (std::size_t i = 1; i < call.args.size(); ++i) {
        analyzeExpr(call.args[i]);
        coerce(call.args[i], elem, call.args[i]->loc, "atomic argument");
        at->operands.push_back(std::move(call.args[i]));
    }
    if (!resultName.empty()) {
        if (resultType && !resultType->isUnknown() && !resultType->sameAs(elem)) {
            diags_.reportError(DiagCode::CGFX0019, loc,
                               "The old value returned by '" + call.callee + "' has type '" + elem.toCppString() +
                                   "': the destination variable must have the same type");
        }
        at->resultName = resultName;
    }
    return at;
}

} // namespace cudagfx

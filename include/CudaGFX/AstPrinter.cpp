// AstPrinter.cpp
#include "AstPrinter.hpp"

#include "Literal.hpp"

namespace cudagfx {

std::string AstPrinter::declarator(const Type& t, const std::string& name) const {
    std::string dims;
    const Type* cur = &t;
    while (cur->isArray() && cur->inner) {
        dims += "[" + std::to_string(cur->arraySize) + "]";
        cur = cur->inner.get();
    }
    return typeName(*cur) + " " + name + dims;
}

void AstPrinter::collectShared(const Stmt& s, std::vector<const DeclStmt::SingleDecl*>& out) {
    switch (s.kind) {
        case StmtKind::Compound:
            for (const auto& c : static_cast<const CompoundStmt&>(s).statements) collectShared(*c, out);
            break;
        case StmtKind::Decl:
            for (const auto& d : static_cast<const DeclStmt&>(s).decls) {
                if (d.isShared) out.push_back(&d);
            }
            break;
        case StmtKind::If: {
            const auto& i = static_cast<const IfStmt&>(s);
            collectShared(*i.thenBranch, out);
            if (i.elseBranch) collectShared(*i.elseBranch, out);
            break;
        }
        case StmtKind::For: {
            const auto& f = static_cast<const ForStmt&>(s);
            if (f.init) collectShared(*f.init, out);
            collectShared(*f.body, out);
            break;
        }
        case StmtKind::While: collectShared(*static_cast<const WhileStmt&>(s).body, out); break;
        case StmtKind::DoWhile: collectShared(*static_cast<const DoWhileStmt&>(s).body, out); break;
        case StmtKind::Switch:
            for (const auto& c : static_cast<const SwitchStmt&>(s).cases) {
                for (const auto& b : c.body) collectShared(*b, out);
            }
            break;
        default:
            break;
    }
}

bool AstPrinter::endsWithJump(const Stmt& s) {
    if (s.kind == StmtKind::Break || s.kind == StmtKind::Continue || s.kind == StmtKind::Return) return true;
    if (s.kind == StmtKind::Compound) {
        const auto& c = static_cast<const CompoundStmt&>(s);
        return !c.statements.empty() && endsWithJump(*c.statements.back());
    }
    return false;
}

// ---------------------------------------------------------------------------------------------------------------
// Statements

void AstPrinter::emitBody(const Stmt& s) {
    out_ << " {\n";
    ++indent_;
    if (s.kind == StmtKind::Compound) {
        for (const auto& c : static_cast<const CompoundStmt&>(s).statements) stmt(*c);
    } else {
        stmt(s);
    }
    --indent_;
    out_ << pad() << "}";
}

void AstPrinter::emitIf(const IfStmt& s, bool continuingElse) {
    std::string cond;
    expr(*s.cond, cond);
    if (!continuingElse) out_ << pad();
    out_ << "if (" << cond << ")";
    emitBody(*s.thenBranch);
    if (s.elseBranch) {
        out_ << " else";
        if (s.elseBranch->kind == StmtKind::If) {
            out_ << ' ';
            emitIf(static_cast<const IfStmt&>(*s.elseBranch), true);
            return;
        }
        emitBody(*s.elseBranch);
    }
    out_ << '\n';
}

void AstPrinter::emitDecl(const DeclStmt& d) {
    for (const auto& sd : d.decls) {
        if (sd.isShared) continue; // hoisted to the module level
        const std::string name = ident(sd.emitName, sd.name);
        if (sd.init && sd.init->kind == ExprKind::Atomic) {
            atomicDeclaration(sd, static_cast<const AtomicExpr&>(*sd.init));
            continue;
        }
        std::string text = std::string(localAttribute()) + declarator(sd.type, name);
        if (sd.init) {
            text += " = ";
            expr(*sd.init, text);
        }
        line(text + ";");
    }
}

void AstPrinter::stmt(const Stmt& s) {
    switch (s.kind) {
        case StmtKind::Compound: {
            line("{");
            ++indent_;
            for (const auto& c : static_cast<const CompoundStmt&>(s).statements) stmt(*c);
            --indent_;
            line("}");
            break;
        }
        case StmtKind::Expr: {
            const Expr& e = *static_cast<const ExprStmt&>(s).expr;
            if (exprStatement(e)) break;
            if (e.kind == ExprKind::Atomic) {
                atomicStatement(static_cast<const AtomicExpr&>(e));
                break;
            }
            // (void)x; : no effect
            if (e.kind == ExprKind::Cast && e.type.isVoid()) {
                const Expr& inner = *static_cast<const CastExpr&>(e).operand;
                if (inner.kind == ExprKind::VarRef || inner.kind == ExprKind::Literal) break;
                std::string t;
                expr(inner, t);
                line(t + ";");
                break;
            }
            std::string t;
            expr(e, t);
            line(t + ";");
            break;
        }
        case StmtKind::Decl:
            emitDecl(static_cast<const DeclStmt&>(s));
            break;
        case StmtKind::If:
            emitIf(static_cast<const IfStmt&>(s), false);
            break;
        case StmtKind::For: {
            const auto& f = static_cast<const ForStmt&>(s);
            std::string init;
            if (f.init) {
                if (f.init->kind == StmtKind::Decl) {
                    const auto& d = static_cast<const DeclStmt&>(*f.init);
                    bool first = true;
                    for (const auto& sd : d.decls) {
                        const std::string name = ident(sd.emitName, sd.name);
                        if (first) init += std::string(localAttribute()) + declarator(sd.type, name);
                        else init += ", " + name;
                        if (sd.init) {
                            init += " = ";
                            expr(*sd.init, init);
                        }
                        first = false;
                    }
                } else if (f.init->kind == StmtKind::Expr) {
                    expr(*static_cast<const ExprStmt&>(*f.init).expr, init);
                }
            }
            std::string cond, inc;
            if (f.cond) expr(*f.cond, cond);
            if (f.inc) expr(*f.inc, inc);
            out_ << pad() << "for (" << init << "; " << cond << "; " << inc << ")";
            emitBody(*f.body);
            out_ << '\n';
            break;
        }
        case StmtKind::While: {
            const auto& w = static_cast<const WhileStmt&>(s);
            std::string cond;
            expr(*w.cond, cond);
            out_ << pad() << "while (" << cond << ")";
            emitBody(*w.body);
            out_ << '\n';
            break;
        }
        case StmtKind::DoWhile: {
            const auto& w = static_cast<const DoWhileStmt&>(s);
            std::string cond;
            expr(*w.cond, cond);
            out_ << pad() << "do";
            emitBody(*w.body);
            out_ << " while (" << cond << ");\n";
            break;
        }
        case StmtKind::Switch: {
            const auto& sw = static_cast<const SwitchStmt&>(s);
            std::string cond;
            expr(*sw.cond, cond);
            line("switch (" + cond + ") {");
            for (std::size_t i = 0; i < sw.cases.size(); ++i) {
                const auto& c = sw.cases[i];
                // `label` was consumed by constant evaluation: the value is in `value`.
                line(c.isDefault ? std::string("default:") : "case " + std::to_string(c.value) + ":");
                ++indent_;
                for (const auto& b : c.body) stmt(*b);
                if (c.body.empty() && i + 1 == sw.cases.size()) line(";");
                // An intentional fall-through into the next label: say so, the generated code stays warning-free.
                if (annotateFallthrough() && i + 1 < sw.cases.size() && !c.body.empty() && !endsWithJump(*c.body.back())) line("[[fallthrough]];");
                --indent_;
            }
            line("}");
            break;
        }
        case StmtKind::Break: line("break;"); break;
        case StmtKind::Continue: line("continue;"); break;
        case StmtKind::Return: {
            const auto& r = static_cast<const ReturnStmt&>(s);
            if (r.value) {
                std::string t;
                expr(*r.value, t);
                line("return " + t + ";");
            } else {
                line("return;");
            }
            break;
        }
        case StmtKind::SyncThreads: line(syncStatement()); break;
        case StmtKind::Empty: break;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Expressions

void AstPrinter::member(const MemberExpr& m, std::string& out, bool lvalue) const {
    expr(*m.base, out, lvalue);
    out += m.viaArrow ? "->" : ".";
    out += fieldName(m);
}

void AstPrinter::index(const IndexExpr& i, std::string& out, bool lvalue) const {
    expr(*i.base, out, lvalue);
    out += '[';
    expr(*i.index, out);
    out += ']';
}

void AstPrinter::expr(const Expr& e, std::string& out, bool lvalue) const {
    switch (e.kind) {
        case ExprKind::Literal: {
            const auto& l = static_cast<const LiteralExpr&>(e);
            switch (l.lit) {
                case LiteralExpr::Lit::Int: out += std::to_string(l.intValue); break;
                case LiteralExpr::Lit::UInt: out += std::to_string(static_cast<std::uint32_t>(l.intValue)) + "u"; break;
                case LiteralExpr::Lit::Float: out += formatFloatLiteral(l.floatValue) + floatSuffix(); break;
                case LiteralExpr::Lit::Bool: out += l.intValue ? "true" : "false"; break;
                case LiteralExpr::Lit::Unsupported: out += "0"; break;
            }
            break;
        }
        case ExprKind::VarRef: varRef(static_cast<const VarRefExpr&>(e), out); break;
        case ExprKind::Builtin: builtin(static_cast<const BuiltinExpr&>(e), out); break;
        case ExprKind::Binary: {
            const auto& b = static_cast<const BinaryExpr&>(e);
            expr(*b.left, out, isAssignmentOp(b.op));
            out += b.op == BinaryOp::Comma ? ", " : std::string(" ") + binaryOpText(b.op) + " ";
            expr(*b.right, out);
            break;
        }
        case ExprKind::Unary: {
            const auto& u = static_cast<const UnaryExpr&>(e);
            std::string operand;
            expr(*u.operand, operand, isIncDecOp(u.op));
            if (isPostfixOp(u.op)) {
                out += operand + unaryOpText(u.op);
            } else {
                const char* op = unaryOpText(u.op);
                out += op;
                // "- -x" must not become "--x"
                if (!operand.empty() && (operand[0] == op[0]) && (op[0] == '-' || op[0] == '+')) out += ' ';
                out += operand;
            }
            break;
        }
        case ExprKind::Paren:
            out += '(';
            expr(*static_cast<const ParenExpr&>(e).sub, out, lvalue);
            out += ')';
            break;
        case ExprKind::Index: index(static_cast<const IndexExpr&>(e), out, lvalue); break;
        case ExprKind::Member: member(static_cast<const MemberExpr&>(e), out, lvalue); break;
        case ExprKind::Call: {
            const auto& c = static_cast<const CallExpr&>(e);
            out += callee(c);
            out += '(';
            for (std::size_t i = 0; i < c.args.size(); ++i) {
                if (i) out += ", ";
                expr(*c.args[i], out);
            }
            out += ')';
            break;
        }
        case ExprKind::Cast: cast(static_cast<const CastExpr&>(e), out); break;
        case ExprKind::Ternary: {
            const auto& t = static_cast<const TernaryExpr&>(e);
            out += '(';
            expr(*t.cond, out);
            out += " ? ";
            expr(*t.thenExpr, out);
            out += " : ";
            expr(*t.elseExpr, out);
            out += ')';
            break;
        }
        case ExprKind::Atomic:
            out += "0"; // unreachable: Sema only allows atomics in statement form
            break;
    }
}

} // namespace cudagfx

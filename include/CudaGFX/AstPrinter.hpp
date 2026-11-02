// AstPrinter.hpp: the code generators' common core.
//
// The shader emitter and the CPU emitter print the same typed AST: statements, control flow, operators, declarations
// are identical, and only a handful of points depend on the target language (identifiers, type names, built-ins,
// casts, atomics, vector swizzles, barriers). AstPrinter owns the shared traversal and leaves exactly those points to
// its subclasses (the hooks below), so there is one place to fix a statement and no way for the two outputs to
// drift apart. Sema has already made every typing decision: nothing here converts or infers anything.
#pragma once

#include <sstream>
#include <string>
#include <vector>
#include "AST.hpp"

namespace cudagfx {

class AstPrinter {
public:
    virtual ~AstPrinter() = default;

protected:
    // ----- output buffer and layout
    std::string pad() const { return std::string(static_cast<std::size_t>(indent_) * 4, ' '); }
    void line(const std::string& text) { out_ << pad() << text << '\n'; }

    // Full declarator: "float tile[16][17]" (dimensions from the outermost to the innermost).
    std::string declarator(const Type& t, const std::string& name) const;

    // The __shared__ declarations of a body, wherever they are (each language hoists them differently).
    static void collectShared(const Stmt& s, std::vector<const DeclStmt::SingleDecl*>& out);

    // ----- shared traversal
    void stmt(const Stmt& s);
    void expr(const Expr& e, std::string& out, bool lvalue = false) const;

    // ----- what a dialect must provide
    virtual std::string typeName(const Type& t) const = 0;
    // Which identifier to print for a symbol: the one chosen by Sema for the shader language (`emitName`) or the
    // one of the source (`sourceName`).
    virtual std::string ident(const std::string& emitName, const std::string& sourceName) const = 0;
    virtual void builtin(const BuiltinExpr& b, std::string& out) const = 0;
    virtual void cast(const CastExpr& c, std::string& out) const = 0;
    virtual std::string syncStatement() const = 0;
    // An atomic used as a statement (`atomicAdd(&a[i], 1);`, or `old = atomicAdd(...)`), and as the initializer of a
    // declaration (`int old = atomicAdd(...)`).
    virtual void atomicStatement(const AtomicExpr& a) = 0;
    virtual void atomicDeclaration(const DeclStmt::SingleDecl& d, const AtomicExpr& a) = 0;

    // ----- what a dialect may change
    virtual const char* floatSuffix() const { return ""; }
    virtual void varRef(const VarRefExpr& v, std::string& out) const { out += ident(v.emitName, v.name); }
    virtual std::string callee(const CallExpr& c) const { return c.emitName; }
    // The member of a struct (or component of a vector) as written in the output: the shader language renames the
    // fields it reserves (`type` -> `cgfx_type`), C++ keeps the source name.
    virtual const std::string& fieldName(const MemberExpr& m) const { return m.emitMember.empty() ? m.member : m.emitMember; }
    virtual void member(const MemberExpr& m, std::string& out, bool lvalue) const;
    virtual void index(const IndexExpr& i, std::string& out, bool lvalue) const;
    virtual const char* localAttribute() const { return ""; }            // before a local declaration
    virtual bool annotateFallthrough() const { return false; }           // `[[fallthrough]];` between cases
    // A dialect may print an expression statement itself (the shader emitter lowers the writes to struct buffers);
    // returns true when it did.
    virtual bool exprStatement(const Expr&) { return false; }

    std::ostringstream out_;
    int indent_ = 0;

private:
    void emitBody(const Stmt& s);
    void emitIf(const IfStmt& s, bool continuingElse);
    void emitDecl(const DeclStmt& d);
    static bool endsWithJump(const Stmt& s);
};

} // namespace cudagfx

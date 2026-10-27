// Sema.hpp: semantic analysis (typing, name resolution, validation of the supported subset).
//
// Fundamental invariant (C4): a wrong shader is never emitted silently. Every construct outside the
// subset produces a CGFXnnnn diagnostic; every implicit conversion becomes an explicit CastExpr in the
// AST, so the emitter no longer has any typing decision to make.
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "AST.hpp"
#include "Diagnostics.hpp"
#include "Intrinsics.hpp"

namespace cudagfx {

class Sema {
public:
    explicit Sema(DiagnosticsEngine& diags) : diags_(diags) {}

    // Analyzes the whole module. Fills in emitName, types, Parameter::access, FunctionDecl::callees,
    // ParsedModule::helperClosure / usedStructs / usedGlobals.
    void analyze(ParsedModule& module);

private:
    enum class Use { Read, Write, ReadWrite };

    struct Symbol {
        SymbolKind kind = SymbolKind::Local;
        Type type;
        std::string emitName;
        bool isConst = false;
        Parameter* param = nullptr;
    };

    struct CallSite {
        const FunctionDecl* callee;
        SourceLocation loc;
    };

    struct GlobalState {
        GlobalConstDecl* decl = nullptr;
        int state = 0; // 0 to analyze, 1 in progress, 2 valid, 3 invalid
    };

    struct StructState {
        StructDecl* decl = nullptr;
        int state = 0;
    };

    struct BinaryPlan {
        bool ok = false;
        Type result;
        Type lhsTarget; // Unknown: no conversion
        Type rhsTarget;
        std::string why;
    };

    // --- declarations
    void indexDeclarations();
    void prepareSignature(FunctionDecl& fn);
    void analyzeFunction(FunctionDecl& fn);
    void classifyBufferAccess(FunctionDecl& fn);
    void buildCallGraph();
    void validateLaunchSites();
    void checkNameCollisions();

    // --- types
    Type resolveType(const Type& t, const SourceLocation& loc);
    void reportUnsupportedType(const Type& t, const SourceLocation& loc);
    bool resolveStruct(const std::string& emitName, const SourceLocation& loc);
    unsigned typeBytes(const Type& t) const;
    const StructDecl* findStruct(const std::string& emitName) const;

    // --- scopes
    void pushScope() { scopes_.emplace_back(); }
    void popScope() { scopes_.pop_back(); }
    bool declare(const std::string& name, const Symbol& sym, const SourceLocation& loc);
    const Symbol* lookup(const std::string& name) const;
    // True if `name` is a global constant usable on the device side (analyzed lazily, silently if it is never used).
    bool lookupGlobalConst(const std::string& name, const SourceLocation& loc, Symbol& out);

    // --- statements
    void analyzeCompound(CompoundStmt& block, bool newScope);
    void analyzeStmt(Stmt& s);
    void analyzeDecl(DeclStmt& decl);
    void analyzeExprStmt(ExprStmt& st);
    void analyzeSubStatement(Stmt& s);
    void analyzeCondition(ExprPtr& cond);

    // --- expressions (SemaExpr.cpp)
    void analyzeExpr(ExprPtr& e, Use use = Use::Read);
    void analyzeLiteral(LiteralExpr& lit);
    void analyzeVarRef(VarRefExpr& ref, Use use, bool pointerOk);
    void analyzeBuiltin(BuiltinExpr& b, Use use);
    void analyzeBinary(ExprPtr& e);
    void analyzeAssignment(ExprPtr& e);
    void analyzeUnary(ExprPtr& e);
    void analyzeIndex(IndexExpr& ix, Use use);
    void analyzeMember(MemberExpr& m, Use use);
    void analyzeCall(CallExpr& c);
    void analyzeUserCall(CallExpr& c, const FunctionDecl& callee);
    void analyzeIntrinsicCall(CallExpr& c, const Intrinsic& in);
    void analyzeCast(CastExpr& c);
    void analyzeTernary(TernaryExpr& t);
    ExprPtr buildAtomic(CallExpr& call, AtomicOp op, const std::string& resultName, const Type* resultType);

    BinaryPlan planBinary(BinaryOp op, const Type& lt, const Type& rt) const;
    Type typeBinaryOp(BinaryOp op, ExprPtr& l, ExprPtr& r, const SourceLocation& loc);
    bool coerce(ExprPtr& e, const Type& target, const SourceLocation& loc, const char* what);
    void toBool(ExprPtr& e);
    void wrapCast(ExprPtr& e, const Type& target, bool implicit);
    ExprPtr makeErrorExpr(const SourceLocation& loc) const;
    bool isLValue(const Expr& e) const;
    bool isPureExpr(const Expr& e) const;
    ExprPtr cloneExpr(const Expr& e) const;

    DiagnosticsEngine& diags_;
    ParsedModule* module_ = nullptr;

    std::unordered_map<std::string, FunctionDecl*> functions_;
    std::unordered_map<std::string, StructState> structs_;
    std::unordered_map<std::string, GlobalState> globals_;
    std::unordered_set<std::string> globalNames_;
    std::unordered_map<const FunctionDecl*, std::vector<CallSite>> callSites_;
    std::unordered_map<const FunctionDecl*, std::unordered_set<std::string>> declaredNames_;

    // context of the function being analyzed
    FunctionDecl* current_ = nullptr;
    std::vector<std::unordered_map<std::string, Symbol>> scopes_;
    std::unordered_set<std::string> sharedEmit_;
    std::unordered_set<std::string> declared_;
    unsigned sharedBytes_ = 0;
    unsigned loopDepth_ = 0;
    unsigned switchDepth_ = 0;
    bool usesGridDim_ = false;
    bool usesPopcount_ = false;
    bool inGlobalInit_ = false;
};

} // namespace cudagfx

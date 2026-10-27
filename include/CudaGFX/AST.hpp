// AST.hpp: the typed abstract syntax tree of CudaGFX.
//
// Every node carries a tag (ExprKind / StmtKind): the passes (Sema, ShaderEmitter) do a `switch`
// followed by a static_cast, with no dynamic_cast chain and no RTTI. Operators are enumerations.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "SourceLocation.hpp"

namespace cudagfx {

// ---------------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------------

struct Type {
    enum class Kind {
        Void, Bool, Int, UInt, Float,
        Float2, Float3, Float4, Int2, Int3, Int4, UInt2, UInt3, UInt4,
        Pointer, Array, Struct,
        Unsupported, // double, long, char, short...: the original C++ name is stored in `name`
        Unknown      // error already reported: suppresses cascading diagnostics
    };

    Kind kind = Kind::Void;
    bool isConst = false;                 // const data (for a pointer: the pointee is const)
    unsigned arraySize = 0;
    std::shared_ptr<const Type> inner;    // pointee, or array element
    std::string name;                     // Struct: emission name; Unsupported: C++ name

    static Type scalar(Kind k);
    static Type makeVoid() { return scalar(Kind::Void); }
    static Type makeBool() { return scalar(Kind::Bool); }
    static Type makeInt() { return scalar(Kind::Int); }
    static Type makeUInt() { return scalar(Kind::UInt); }
    static Type makeFloat() { return scalar(Kind::Float); }
    static Type makeUnknown() { return scalar(Kind::Unknown); }
    static Type makeVector(Kind scalarKind, unsigned components);
    static Type makePointer(const Type& pointee, bool pointeeConst);
    static Type makeArray(const Type& element, unsigned size);
    static Type makeStruct(std::string emitName);
    static Type makeUnsupported(std::string cppName);

    bool isVoid() const { return kind == Kind::Void; }
    bool isScalar() const { return kind == Kind::Bool || kind == Kind::Int || kind == Kind::UInt || kind == Kind::Float; }
    bool isVector() const { return kind >= Kind::Float2 && kind <= Kind::UInt4; }
    bool isInteger() const { return kind == Kind::Int || kind == Kind::UInt; }
    bool isFloating() const { return kind == Kind::Float; }
    bool isPointer() const { return kind == Kind::Pointer; }
    bool isArray() const { return kind == Kind::Array; }
    bool isStruct() const { return kind == Kind::Struct; }
    bool isUnsupported() const { return kind == Kind::Unsupported; }
    bool isUnknown() const { return kind == Kind::Unknown; }

    // Scalar type of the components of a vector (or the scalar itself); Unknown otherwise.
    Kind scalarKind() const;
    // 1 for a scalar, 2..4 for a vector, 0 otherwise.
    unsigned componentCount() const;
    // Structural equality (the top-level const qualifier is ignored).
    bool sameAs(const Type& other) const;

    std::string toShaderString() const;  // element type for an array
    std::string toCppString() const;
    // Estimated size in shared memory (vec3 counts for 16 bytes: conservative layout).
    unsigned sizeInBytes() const;
};

// ---------------------------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------------------------

enum class BinaryOp {
    Add, Sub, Mul, Div, Mod, Shl, Shr, BitAnd, BitOr, BitXor,
    LogAnd, LogOr, Eq, Ne, Lt, Le, Gt, Ge,
    Assign, AddAssign, SubAssign, MulAssign, DivAssign, ModAssign,
    ShlAssign, ShrAssign, AndAssign, OrAssign, XorAssign,
    Comma
};
const char* binaryOpText(BinaryOp op);
bool isAssignmentOp(BinaryOp op);
bool isCompoundAssignmentOp(BinaryOp op);
bool isComparisonOp(BinaryOp op);
bool isLogicalOp(BinaryOp op);
// `x op= y` -> arithmetic operator `op`.
BinaryOp compoundBase(BinaryOp op);

enum class UnaryOp { Plus, Minus, LogNot, BitNot, PreInc, PreDec, PostInc, PostDec, Deref, AddressOf };
const char* unaryOpText(UnaryOp op);
inline bool isPostfixOp(UnaryOp op) { return op == UnaryOp::PostInc || op == UnaryOp::PostDec; }
inline bool isIncDecOp(UnaryOp op) {
    return op == UnaryOp::PreInc || op == UnaryOp::PreDec || op == UnaryOp::PostInc || op == UnaryOp::PostDec;
}

// ---------------------------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------------------------

struct ASTNode {
    SourceLocation loc;
    virtual ~ASTNode() = default;
};

enum class ExprKind { Literal, VarRef, Builtin, Binary, Unary, Paren, Index, Member, Call, Cast, Ternary, Atomic };

struct Expr : ASTNode {
    explicit Expr(ExprKind k) : kind(k) {}
    ExprKind kind;
    Type type;
};
using ExprPtr = std::unique_ptr<Expr>;

struct LiteralExpr : Expr {
    enum class Lit { Int, UInt, Float, Bool, Unsupported };
    LiteralExpr() : Expr(ExprKind::Literal) {}
    Lit lit = Lit::Int;
    std::int64_t intValue = 0;
    double floatValue = 0.0;
    std::string text; // source text (for messages)
};

enum class SymbolKind { Unresolved, ScalarParam, BufferParam, Local, Shared, GlobalConst };

struct VarRefExpr : Expr {
    VarRefExpr() : Expr(ExprKind::VarRef) {}
    std::string name;     // name in the source
    std::string emitName; // name in the shader (after reserved-word protection)
    SymbolKind symbol = SymbolKind::Unresolved;
    bool isConstData = false; // const* buffer, constant, const variable: not assignable
};

struct BuiltinExpr : Expr {
    enum class Kind { ThreadIdx, BlockIdx, BlockDim, GridDim };
    BuiltinExpr() : Expr(ExprKind::Builtin) {}
    Kind builtin = Kind::ThreadIdx;
    char component = 0; // 'x', 'y', 'z'; 0 = the whole vector (uint3)
};

struct BinaryExpr : Expr {
    BinaryExpr() : Expr(ExprKind::Binary) {}
    BinaryOp op = BinaryOp::Add;
    ExprPtr left, right;
};

struct UnaryExpr : Expr {
    UnaryExpr() : Expr(ExprKind::Unary) {}
    UnaryOp op = UnaryOp::Plus;
    ExprPtr operand;
};

struct ParenExpr : Expr {
    ParenExpr() : Expr(ExprKind::Paren) {}
    ExprPtr sub;
};

struct IndexExpr : Expr {
    IndexExpr() : Expr(ExprKind::Index) {}
    ExprPtr base, index;
};

struct MemberExpr : Expr {
    MemberExpr() : Expr(ExprKind::Member) {}
    ExprPtr base;
    std::string member;
    bool viaArrow = false;
    std::string emitMember; // emission name for struct fields
};

struct FunctionDecl;
struct Intrinsic;

struct CallExpr : Expr {
    CallExpr() : Expr(ExprKind::Call) {}
    std::string callee;
    std::vector<ExprPtr> args;
    const FunctionDecl* target = nullptr;   // the user's __device__ function
    const Intrinsic* intrinsic = nullptr;   // math function / vector constructor
    std::string emitName;
};

struct CastExpr : Expr {
    CastExpr() : Expr(ExprKind::Cast) {}
    ExprPtr operand;
    bool implicit = false;
};

struct TernaryExpr : Expr {
    TernaryExpr() : Expr(ExprKind::Ternary) {}
    ExprPtr cond, thenExpr, elseExpr;
};

enum class AtomicOp { Add, Sub, Min, Max, And, Or, Xor, Exch, Cas };

// Atomic validated by Sema (statement form only, see CGFX0015). An empty `resultName` = result ignored.
struct AtomicExpr : Expr {
    AtomicExpr() : Expr(ExprKind::Atomic) {}
    AtomicOp op = AtomicOp::Add;
    ExprPtr target;                 // the targeted element: a[i]
    std::vector<ExprPtr> operands;  // value; (comparand, value) for CAS
    std::string resultName;         // variable receiving the old value
};

// ---------------------------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------------------------

enum class StmtKind { Compound, Expr, Decl, If, For, While, DoWhile, Switch, Break, Continue, Return, SyncThreads, Empty };

struct Stmt : ASTNode {
    explicit Stmt(StmtKind k) : kind(k) {}
    StmtKind kind;
};
using StmtPtr = std::unique_ptr<Stmt>;

struct CompoundStmt : Stmt {
    CompoundStmt() : Stmt(StmtKind::Compound) {}
    std::vector<StmtPtr> statements;
    SourceLocation endLoc; // position of the closing brace
};

struct ExprStmt : Stmt {
    ExprStmt() : Stmt(StmtKind::Expr) {}
    ExprPtr expr;
};

struct DeclStmt : Stmt {
    DeclStmt() : Stmt(StmtKind::Decl) {}
    struct SingleDecl {
        Type type;
        std::string name;
        std::string emitName;
        ExprPtr init;
        bool isShared = false;
        bool isConst = false;
        SourceLocation loc;
    };
    std::vector<SingleDecl> decls;
};

struct IfStmt : Stmt {
    IfStmt() : Stmt(StmtKind::If) {}
    ExprPtr cond;
    StmtPtr thenBranch, elseBranch;
};

struct ForStmt : Stmt {
    ForStmt() : Stmt(StmtKind::For) {}
    StmtPtr init; // DeclStmt or ExprStmt, possibly null
    ExprPtr cond, inc;
    StmtPtr body;
};

struct WhileStmt : Stmt {
    WhileStmt() : Stmt(StmtKind::While) {}
    ExprPtr cond;
    StmtPtr body;
};

struct DoWhileStmt : Stmt {
    DoWhileStmt() : Stmt(StmtKind::DoWhile) {}
    StmtPtr body;
    ExprPtr cond;
};

struct SwitchCase {
    bool isDefault = false;
    ExprPtr label;
    std::int64_t value = 0;
    std::vector<StmtPtr> body;
    SourceLocation loc;
};

struct SwitchStmt : Stmt {
    SwitchStmt() : Stmt(StmtKind::Switch) {}
    ExprPtr cond;
    std::vector<SwitchCase> cases;
};

struct BreakStmt : Stmt { BreakStmt() : Stmt(StmtKind::Break) {} };
struct ContinueStmt : Stmt { ContinueStmt() : Stmt(StmtKind::Continue) {} };
struct SyncThreadsStmt : Stmt { SyncThreadsStmt() : Stmt(StmtKind::SyncThreads) {} };
struct EmptyStmt : Stmt { EmptyStmt() : Stmt(StmtKind::Empty) {} };

struct ReturnStmt : Stmt {
    ReturnStmt() : Stmt(StmtKind::Return) {}
    ExprPtr value;
};

// ---------------------------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------------------------

struct Parameter {
    enum class Access { None, ReadOnly, WriteOnly, ReadWrite };
    Type type;
    std::string name;
    std::string emitName;
    SourceLocation loc;
    // Filled in by Sema for pointer parameters (buffers).
    Access access = Access::None;
    bool isRead = false;
    bool isWritten = false;
};

struct FunctionDecl : ASTNode {
    std::string name;
    bool isKernel = false;
    bool isDevice = false;
    bool isHost = false;
    unsigned launchBounds = 0;
    Type returnType;
    std::vector<Parameter> parameters;
    std::unique_ptr<CompoundStmt> body; // null: simple prototype
    SourceSpan span;                    // from the first qualifier to `}` or `;`

    // Results of Sema
    bool usesGridDim = false;
    bool usesPopcount = false;   // calls __popc: the shader needs the cgfx_popcount macro
    std::vector<const FunctionDecl*> callees;
    unsigned sharedBytes = 0;
};

struct StructField {
    Type type;
    std::string name;
    std::string emitName;
};

struct StructDecl {
    std::string name;
    std::string emitName;
    std::vector<StructField> fields;
    SourceLocation loc;
};

struct GlobalConstDecl {
    Type type;
    std::string name;
    std::string emitName;
    ExprPtr init;
    SourceLocation loc;       // position of the name
    SourceLocation declStart; // start of the declaration (first qualifier), used when rewriting the host code
};

struct LaunchSite {
    std::string kernelName;
    SourceLocation kernelLoc;
    SourceSpan span;                       // from `kernel` up to and including `)`
    std::string gridExprText, blockExprText;
    std::vector<std::string> argExprTexts;
    unsigned evaluatedBlock[3] = {1, 1, 1};
    bool inMainFile = true;
};

struct ParsedModule {
    FileId mainFile = kNoFile;
    std::vector<std::unique_ptr<FunctionDecl>> functions;
    std::vector<std::unique_ptr<StructDecl>> structs;
    std::vector<std::unique_ptr<GlobalConstDecl>> globals;
    std::vector<LaunchSite> launchSites;
    // Filled in by Sema: for each kernel, the __device__ functions it calls, callees before callers.
    std::unordered_map<const FunctionDecl*, std::vector<const FunctionDecl*>> helperClosure;
    // Filled in by Sema: structs and global constants actually used by device code,
    // in an order where each element only depends on the previous ones.
    std::vector<const StructDecl*> usedStructs;
    std::vector<const GlobalConstDecl*> usedGlobals;

    const FunctionDecl* findFunction(const std::string& name) const;
};

} // namespace cudagfx

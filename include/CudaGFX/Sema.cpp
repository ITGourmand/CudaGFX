// Sema.cpp: declarations, types, statements, call graph, launch-site validation.
#include "Sema.hpp"

#include <algorithm>
#include <functional>
#include "Intrinsics.hpp"
#include "ShaderNames.hpp"

namespace cudagfx {

namespace {

constexpr unsigned kMaxBlockX = 1024;
constexpr unsigned kMaxBlockY = 1024;
constexpr unsigned kMaxBlockZ = 64;
constexpr unsigned kMaxThreadsPerBlock = 1024;
constexpr unsigned kMaxSharedBytes = 32768; // ceiling of the backends (D3D11/Metal); beyond it: error
constexpr unsigned kPortableSharedBytes = 16384; // minimum guaranteed by Vulkan/WebGPU; beyond it: warning

// C/C++ types known to CudaGFX but outside the supported 32-bit subset (CGFX0002 / CGFX0012).
bool isKnownUnsupportedTypeName(const std::string& n) {
    static const std::unordered_set<std::string> known = {
        "double", "long", "long long", "long double", "char", "short", "unsigned char", "signed char",
        "unsigned short", "unsigned long", "size_t", "ssize_t", "ptrdiff_t", "int8_t", "int16_t", "int64_t",
        "uint8_t", "uint16_t", "uint64_t", "half", "__half", "half2", "dim3", "float16_t"};
    return known.count(n) != 0;
}

bool isBufferElementKind(Type::Kind k) {
    switch (k) {
        case Type::Kind::Float: case Type::Kind::Float2: case Type::Kind::Float4:
        case Type::Kind::Int: case Type::Kind::Int2: case Type::Kind::Int4:
        case Type::Kind::UInt: case Type::Kind::UInt2: case Type::Kind::UInt4:
            return true;
        default:
            return false;
    }
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------------------------

void Sema::analyze(ParsedModule& module) {
    module_ = &module;
    indexDeclarations();
    for (auto& f : module.functions) {
        if (f->isKernel || f->isDevice) prepareSignature(*f);
    }
    for (auto& f : module.functions) {
        if ((f->isKernel || f->isDevice) && f->body) analyzeFunction(*f);
    }
    buildCallGraph();
    validateLaunchSites();
    checkNameCollisions();
}

void Sema::indexDeclarations() {
    for (auto& f : module_->functions) {
        globalNames_.insert(f->name);
        auto it = functions_.find(f->name);
        if (it == functions_.end()) {
            functions_[f->name] = f.get();
            continue;
        }
        FunctionDecl* prev = it->second;
        bool sameSignature = prev->parameters.size() == f->parameters.size() && prev->returnType.sameAs(f->returnType);
        for (std::size_t i = 0; sameSignature && i < prev->parameters.size(); ++i) {
            sameSignature = prev->parameters[i].type.sameAs(f->parameters[i].type);
        }
        if (!sameSignature) {
            diags_.reportError(DiagCode::CGFX0019, f->loc,
                               "Overloading of function '" + f->name + "' is not supported: a name can only designate a single signature");
        } else if (prev->body && f->body) {
            diags_.reportError(DiagCode::CGFX0019, f->loc, "Redefinition of function '" + f->name + "'");
        } else if (!prev->body && f->body) {
            it->second = f.get();
        }
    }
    for (auto& s : module_->structs) {
        globalNames_.insert(s->name);
        structs_[s->emitName].decl = s.get();
    }
    for (auto& g : module_->globals) {
        globalNames_.insert(g->name);
        globals_[g->name].decl = g.get();
    }
}

// ---------------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------------

void Sema::reportUnsupportedType(const Type& t, const SourceLocation& loc) {
    if (isKnownUnsupportedTypeName(t.name)) {
        diags_.reportError(DiagCode::CGFX0002, loc,
                           "Unsupported type: '" + t.name +
                               "' (supported types: bool, int, unsigned int, float, float2/3/4, int2/3/4, uint2/3/4)");
    } else {
        diags_.reportError(DiagCode::CGFX0010, loc,
                           "Unknown type: '" + t.name +
                               "' (typedef/using aliases are not supported: use the native type or #define)");
    }
}

const StructDecl* Sema::findStruct(const std::string& emitName) const {
    const auto it = structs_.find(emitName);
    return it == structs_.end() ? nullptr : it->second.decl;
}

bool Sema::resolveStruct(const std::string& emitName, const SourceLocation& loc) {
    auto it = structs_.find(emitName);
    if (it == structs_.end() || !it->second.decl) {
        diags_.reportError(DiagCode::CGFX0010, loc, "Unknown type: struct '" + emitName + "'");
        return false;
    }
    StructState& st = it->second;
    if (st.state == 2) return true;
    if (st.state == 3) return false;
    if (st.state == 1) {
        diags_.reportError(DiagCode::CGFX0019, loc, "Recursive struct: '" + st.decl->name + "'");
        return false;
    }
    st.state = 1;
    bool ok = true;
    for (auto& f : st.decl->fields) {
        if (isForbiddenIdentifier(f.name)) {
            diags_.reportError(DiagCode::CGFX0016, st.decl->loc, "Reserved field name: '" + f.name + "'");
            ok = false;
        }
        Type base = f.type;
        while (base.isArray() && base.inner) base = *base.inner;
        if (base.isPointer()) {
            diags_.reportError(DiagCode::CGFX0001, st.decl->loc,
                               "Pointer field '" + f.name + "' in struct '" + st.decl->name + "': not supported");
            ok = false;
            continue;
        }
        const Type r = resolveType(f.type, st.decl->loc);
        if (r.isUnknown()) {
            ok = false;
        } else {
            f.type = r;
        }
    }
    st.state = ok ? 2 : 3;
    if (ok) module_->usedStructs.push_back(st.decl);
    return ok;
}

Type Sema::resolveType(const Type& t, const SourceLocation& loc) {
    switch (t.kind) {
        case Type::Kind::Unsupported:
            reportUnsupportedType(t, loc);
            return Type::makeUnknown();
        case Type::Kind::Struct:
            return resolveStruct(t.name, loc) ? t : Type::makeUnknown();
        case Type::Kind::Pointer: {
            if (!t.inner) return Type::makeUnknown();
            const Type inner = resolveType(*t.inner, loc);
            if (inner.isUnknown()) return Type::makeUnknown();
            return Type::makePointer(inner, t.isConst);
        }
        case Type::Kind::Array: {
            if (!t.inner) return Type::makeUnknown();
            const Type inner = resolveType(*t.inner, loc);
            if (inner.isUnknown()) return Type::makeUnknown();
            Type r = Type::makeArray(inner, t.arraySize);
            r.isConst = t.isConst;
            return r;
        }
        default:
            return t;
    }
}

unsigned Sema::typeBytes(const Type& t) const {
    switch (t.kind) {
        case Type::Kind::Array: return t.inner ? t.arraySize * typeBytes(*t.inner) : 0;
        case Type::Kind::Struct: {
            const StructDecl* s = findStruct(t.name);
            unsigned total = 0;
            if (s) {
                for (const auto& f : s->fields) total += typeBytes(f.type);
            }
            return total;
        }
        default: return t.sizeInBytes();
    }
}

// ---------------------------------------------------------------------------------------------
// Scopes
// ---------------------------------------------------------------------------------------------

bool Sema::declare(const std::string& name, const Symbol& sym, const SourceLocation& loc) {
    if (scopes_.empty()) pushScope();
    auto& scope = scopes_.back();
    if (scope.count(name)) {
        diags_.reportError(DiagCode::CGFX0019, loc, "Redeclaration of '" + name + "' in the same scope");
        return false;
    }
    scope[name] = sym;
    return true;
}

const Sema::Symbol* Sema::lookup(const std::string& name) const {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
        const auto f = it->find(name);
        if (f != it->end()) return &f->second;
    }
    return nullptr;
}

bool Sema::lookupGlobalConst(const std::string& name, const SourceLocation& loc, Symbol& out) {
    const auto it = globals_.find(name);
    if (it == globals_.end() || !it->second.decl) return false;
    GlobalState& gs = it->second;
    auto fill = [&]() {
        out = Symbol();
        out.kind = SymbolKind::GlobalConst;
        out.type = gs.decl->type;
        out.emitName = gs.decl->emitName;
        out.isConst = true;
    };
    if (gs.state == 2) {
        fill();
        return true;
    }
    if (gs.state == 1) {
        diags_.reportError(DiagCode::CGFX0019, loc, "Global constant '" + name + "' is defined in terms of itself");
        return false;
    }
    if (gs.state == 3) {
        diags_.reportError(DiagCode::CGFX0010, loc,
                           "Global constant '" + name + "' cannot be used in device code: "
                           "its initializer is not a supported expression");
        return false;
    }

    // Lazy and silent analysis: an arbitrary host constant must not make the compilation fail as long as
    // no device code uses it.
    gs.state = 1;
    FunctionDecl* savedFn = current_;
    auto savedScopes = std::move(scopes_);
    scopes_.clear();
    const bool savedGlobalInit = inGlobalInit_;
    inGlobalInit_ = true;
    diags_.setMuted(true);
    diags_.resetMutedErrors();
    gs.decl->type = resolveType(gs.decl->type, gs.decl->loc);
    bool ok = !gs.decl->type.isUnknown();
    if (ok) {
        analyzeExpr(gs.decl->init);
        ok = !gs.decl->init->type.isUnknown() && coerce(gs.decl->init, gs.decl->type, gs.decl->loc, "initialization");
    }
    ok = ok && diags_.mutedErrorCount() == 0;
    diags_.setMuted(false);
    diags_.resetMutedErrors();
    inGlobalInit_ = savedGlobalInit;
    scopes_ = std::move(savedScopes);
    current_ = savedFn;

    if (!ok) {
        gs.state = 3;
        return lookupGlobalConst(name, loc, out);
    }
    gs.decl->emitName = safeIdentifier(gs.decl->name);
    gs.state = 2;
    module_->usedGlobals.push_back(gs.decl);
    fill();
    return true;
}

// ---------------------------------------------------------------------------------------------
// Signatures and functions
// ---------------------------------------------------------------------------------------------

void Sema::prepareSignature(FunctionDecl& fn) {
    if (isForbiddenIdentifier(fn.name)) {
        diags_.reportError(DiagCode::CGFX0016, fn.loc, "Reserved function name: '" + fn.name + "'");
    }
    if (fn.isKernel && fn.name == "launch") {
        diags_.reportError(DiagCode::CGFX0016, fn.loc, "Reserved kernel name: 'launch' (name of the generated stub function)");
    }
    fn.returnType = resolveType(fn.returnType, fn.loc);
    if (fn.isKernel) {
        if (!fn.returnType.isVoid() && !fn.returnType.isUnknown()) {
            diags_.reportError(DiagCode::CGFX0019, fn.loc, "A __global__ kernel must return void");
        }
    } else if (fn.returnType.isPointer()) {
        diags_.reportError(DiagCode::CGFX0001, fn.loc, "A __device__ function cannot return a pointer");
    }

    for (std::size_t i = 0; i < fn.parameters.size(); ++i) {
        Parameter& p = fn.parameters[i];
        p.emitName = p.name.empty() ? "cgfx_p" + std::to_string(i) : safeIdentifier(p.name);
        if (!p.name.empty() && isForbiddenIdentifier(p.name)) {
            diags_.reportError(DiagCode::CGFX0016, p.loc, "Reserved parameter name: '" + p.name + "'");
        }
        if (p.type.isPointer()) {
            const Type* inner = p.type.inner.get();
            if (!fn.isKernel) {
                diags_.reportError(DiagCode::CGFX0001, p.loc,
                                   "Pointer parameter in __device__ function '" + fn.name +
                                       "' is not supported: buffers can only be passed to kernels (pass the index or the value)");
                p.type = Type::makeUnknown();
            } else if (!inner) {
                p.type = Type::makeUnknown();
            } else if (inner->isPointer()) {
                diags_.reportError(DiagCode::CGFX0001, p.loc, "Pointer to pointer is not supported for '" + p.name + "'");
                p.type = Type::makeUnknown();
            } else if (inner->isUnsupported() && !isKnownUnsupportedTypeName(inner->name)) {
                reportUnsupportedType(*inner, p.loc); // unknown type name (alias?): CGFX0010 with a hint
                p.type = Type::makeUnknown();
            } else if (inner->isUnsupported() || inner->isStruct() || !isBufferElementKind(inner->kind)) {
                diags_.reportError(DiagCode::CGFX0012, p.loc,
                                   "Unsupported buffer element type: '" + p.type.toCppString() +
                                       "' (supported elements: float, int, unsigned int, float2/4, int2/4, uint2/4; "
                                       "for 3 components, use the 4-component type)");
                p.type = Type::makeUnknown();
            }
            continue;
        }
        const Type resolved = resolveType(p.type, p.loc);
        if (fn.isKernel && !resolved.isUnknown() && !(resolved.isScalar() || resolved.isVector())) {
            diags_.reportError(DiagCode::CGFX0002, p.loc,
                               "Unsupported kernel parameter type: '" + resolved.toCppString() +
                                   "' (scalars, vectors and buffer pointers only)");
            p.type = Type::makeUnknown();
        } else if (!resolved.isUnknown() && resolved.isVoid()) {
            diags_.reportError(DiagCode::CGFX0019, p.loc, "Parameter of type void");
            p.type = Type::makeUnknown();
        } else if (!resolved.isUnknown() && resolved.isArray()) {
            p.type = Type::makeUnknown();
        } else {
            p.type = resolved;
        }
    }
}

void Sema::analyzeFunction(FunctionDecl& fn) {
    current_ = &fn;
    scopes_.clear();
    pushScope();
    sharedEmit_.clear();
    declared_.clear();
    sharedBytes_ = 0;
    loopDepth_ = 0;
    switchDepth_ = 0;
    usesGridDim_ = false;
    usesPopcount_ = false;

    for (auto& p : fn.parameters) {
        Symbol s;
        s.type = p.type;
        s.emitName = p.emitName;
        s.param = &p;
        if (p.type.isPointer()) {
            s.kind = SymbolKind::BufferParam;
            s.isConst = p.type.isConst;
        } else {
            s.kind = fn.isKernel ? SymbolKind::ScalarParam : SymbolKind::Local;
        }
        if (!p.name.empty()) {
            declared_.insert(p.name);
            declare(p.name, s, p.loc);
        }
    }

    analyzeCompound(*fn.body, true);

    fn.sharedBytes = sharedBytes_;
    fn.usesGridDim = usesGridDim_;
    fn.usesPopcount = usesPopcount_;
    if (fn.isKernel) {
        if (sharedBytes_ > kMaxSharedBytes) {
            diags_.reportError(DiagCode::CGFX0008, fn.loc,
                               "Shared memory of kernel '" + fn.name + "' : " + std::to_string(sharedBytes_) +
                                   " bytes, beyond the limit of " + std::to_string(kMaxSharedBytes));
        } else if (sharedBytes_ > kPortableSharedBytes) {
            diags_.reportWarning(DiagCode::CGFX0008, fn.loc,
                                 "Shared memory of kernel '" + fn.name + "' : " + std::to_string(sharedBytes_) +
                                     " bytes, beyond " + std::to_string(kPortableSharedBytes) +
                                     " (minimum guaranteed by Vulkan/WebGPU): may fail on some GPUs");
        }
    }
    classifyBufferAccess(fn);
    declaredNames_[&fn] = declared_;
    scopes_.clear();
    current_ = nullptr;
}

void Sema::classifyBufferAccess(FunctionDecl& fn) {
    for (auto& p : fn.parameters) {
        if (!p.type.isPointer()) continue;
        if (p.type.isConst) p.access = Parameter::Access::ReadOnly;
        else if (!p.isWritten) p.access = p.isRead ? Parameter::Access::ReadOnly : Parameter::Access::ReadWrite;
        else p.access = p.isRead ? Parameter::Access::ReadWrite : Parameter::Access::WriteOnly;
    }
}

// ---------------------------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------------------------

void Sema::analyzeCompound(CompoundStmt& block, bool newScope) {
    if (newScope) pushScope();
    for (auto& s : block.statements) {
        if (s) analyzeStmt(*s);
    }
    if (newScope) popScope();
}

void Sema::analyzeSubStatement(Stmt& s) {
    pushScope();
    analyzeStmt(s);
    popScope();
}

void Sema::analyzeCondition(ExprPtr& cond) {
    analyzeExpr(cond);
    toBool(cond);
}

void Sema::analyzeStmt(Stmt& s) {
    switch (s.kind) {
        case StmtKind::Compound:
            analyzeCompound(static_cast<CompoundStmt&>(s), true);
            break;
        case StmtKind::Expr:
            analyzeExprStmt(static_cast<ExprStmt&>(s));
            break;
        case StmtKind::Decl:
            analyzeDecl(static_cast<DeclStmt&>(s));
            break;
        case StmtKind::If: {
            auto& st = static_cast<IfStmt&>(s);
            analyzeCondition(st.cond);
            analyzeSubStatement(*st.thenBranch);
            if (st.elseBranch) analyzeSubStatement(*st.elseBranch);
            break;
        }
        case StmtKind::For: {
            auto& st = static_cast<ForStmt&>(s);
            pushScope();
            if (st.init) {
                analyzeStmt(*st.init);
                bool atomicInit = false;
                if (st.init->kind == StmtKind::Decl) {
                    for (const auto& d : static_cast<DeclStmt&>(*st.init).decls) {
                        if (d.init && d.init->kind == ExprKind::Atomic) atomicInit = true;
                    }
                } else if (st.init->kind == StmtKind::Expr) {
                    const auto& e = static_cast<ExprStmt&>(*st.init).expr;
                    if (e && e->kind == ExprKind::Atomic) atomicInit = true;
                }
                if (atomicInit) {
                    diags_.reportError(DiagCode::CGFX0015, st.init->loc,
                                       "Atomic in the initialization of a for: not supported (move it out of the loop)");
                }
            }
            if (st.cond) analyzeCondition(st.cond);
            if (st.inc) analyzeExpr(st.inc);
            ++loopDepth_;
            analyzeSubStatement(*st.body);
            --loopDepth_;
            popScope();
            break;
        }
        case StmtKind::While: {
            auto& st = static_cast<WhileStmt&>(s);
            analyzeCondition(st.cond);
            ++loopDepth_;
            analyzeSubStatement(*st.body);
            --loopDepth_;
            break;
        }
        case StmtKind::DoWhile: {
            auto& st = static_cast<DoWhileStmt&>(s);
            ++loopDepth_;
            analyzeSubStatement(*st.body);
            --loopDepth_;
            analyzeCondition(st.cond);
            break;
        }
        case StmtKind::Switch: {
            auto& st = static_cast<SwitchStmt&>(s);
            analyzeExpr(st.cond);
            if (!st.cond->type.isUnknown()) {
                if (st.cond->type.kind == Type::Kind::Bool) wrapCast(st.cond, Type::makeInt(), true);
                else if (!st.cond->type.isInteger()) {
                    diags_.reportError(DiagCode::CGFX0019, st.cond->loc,
                                       "The expression of a switch must be an integer (type " + st.cond->type.toCppString() + ")");
                }
            }
            ++switchDepth_;
            pushScope();
            std::unordered_set<std::int64_t> seen;
            bool sawDefault = false;
            for (auto& c : st.cases) {
                if (c.isDefault) {
                    if (sawDefault) diags_.reportError(DiagCode::CGFX0019, c.loc, "Multiple 'default' labels in the same switch");
                    sawDefault = true;
                } else if (!seen.insert(c.value).second) {
                    diags_.reportError(DiagCode::CGFX0019, c.loc, "Duplicate case value: " + std::to_string(c.value));
                }
                for (auto& b : c.body) {
                    if (b) analyzeStmt(*b);
                }
            }
            popScope();
            --switchDepth_;
            break;
        }
        case StmtKind::Break:
            if (loopDepth_ == 0 && switchDepth_ == 0) diags_.reportError(DiagCode::CGFX0019, s.loc, "'break' outside of a loop or a switch");
            break;
        case StmtKind::Continue:
            if (loopDepth_ == 0) diags_.reportError(DiagCode::CGFX0019, s.loc, "'continue' outside of a loop");
            break;
        case StmtKind::Return: {
            auto& st = static_cast<ReturnStmt&>(s);
            if (st.value) analyzeExpr(st.value);
            if (current_->isKernel) {
                if (st.value) diags_.reportError(DiagCode::CGFX0019, s.loc, "A kernel does not return a value");
            } else if (st.value) {
                if (current_->returnType.isVoid()) {
                    diags_.reportError(DiagCode::CGFX0019, s.loc, "'return' with a value in a void function");
                } else {
                    coerce(st.value, current_->returnType, s.loc, "return");
                }
            } else if (!current_->returnType.isVoid() && !current_->returnType.isUnknown()) {
                diags_.reportError(DiagCode::CGFX0019, s.loc, "'return' without a value in a non-void function");
            }
            break;
        }
        case StmtKind::SyncThreads:
        case StmtKind::Empty:
            break;
    }
}

void Sema::analyzeExprStmt(ExprStmt& st) {
    Expr* e = st.expr.get();
    AtomicOp op;
    // Atomic whose result is ignored: atomicAdd(&a[i], v);
    if (e->kind == ExprKind::Call && parseAtomicName(static_cast<CallExpr*>(e)->callee, op)) {
        st.expr = buildAtomic(*static_cast<CallExpr*>(e), op, "", nullptr);
        return;
    }
    // Atomic whose result is assigned: old = atomicAdd(&a[i], v);
    if (e->kind == ExprKind::Binary) {
        auto* b = static_cast<BinaryExpr*>(e);
        if (b->op == BinaryOp::Assign && b->right->kind == ExprKind::Call &&
            parseAtomicName(static_cast<CallExpr*>(b->right.get())->callee, op)) {
            analyzeExpr(b->left, Use::Write);
            if (b->left->kind != ExprKind::VarRef || !isLValue(*b->left)) {
                diags_.reportError(DiagCode::CGFX0015, b->loc,
                                   "The result of an atomic can only be assigned to a plain local variable");
                return;
            }
            auto* ref = static_cast<VarRefExpr*>(b->left.get());
            Type t = ref->type;
            const std::string resultName = ref->emitName;
            st.expr = buildAtomic(*static_cast<CallExpr*>(b->right.get()), op, resultName, &t);
            return;
        }
    }
    analyzeExpr(st.expr);
}

void Sema::analyzeDecl(DeclStmt& ds) {
    for (auto& d : ds.decls) {
        if (isForbiddenIdentifier(d.name)) {
            diags_.reportError(DiagCode::CGFX0016, d.loc,
                               "Reserved variable name: '" + d.name + "' (the gl_, bgfx_, cgfx_ prefixes and double underscores are forbidden)");
        }
        d.type = resolveType(d.type, d.loc);
        bool bad = d.type.isUnknown();
        if (!bad) {
            Type base = d.type;
            while (base.isArray() && base.inner) base = *base.inner;
            if (base.isPointer()) {
                diags_.reportError(DiagCode::CGFX0001, d.loc,
                                   "Local variable of pointer type ('" + d.name +
                                       "') is not supported: pointers are only used to index the buffers passed to the kernel");
                bad = true;
            } else if (base.isVoid()) {
                diags_.reportError(DiagCode::CGFX0019, d.loc, "Variable '" + d.name + "' of type void");
                bad = true;
            }
        }
        if (bad) d.type = Type::makeUnknown();

        Symbol sym;
        sym.type = d.type;
        sym.isConst = d.isConst;
        if (d.isShared) {
            sym.kind = SymbolKind::Shared;
            if (!current_->isKernel) {
                diags_.reportError(DiagCode::CGFX0009, d.loc, "__shared__ variable in a __device__ function is not supported: declare it in the kernel");
            }
            std::string candidate = safeIdentifier(d.name);
            const std::string base = candidate;
            unsigned n = 1;
            auto taken = [&](const std::string& c) {
                if (globalNames_.count(c) || sharedEmit_.count(c)) return true;
                for (const auto& p : current_->parameters) {
                    if (p.emitName == c) return true;
                }
                return false;
            };
            while (taken(candidate)) candidate = base + "_" + std::to_string(n++);
            sharedEmit_.insert(candidate);
            sym.emitName = candidate;
            if (!bad) sharedBytes_ += typeBytes(d.type);
            if (d.init) {
                diags_.reportError(DiagCode::CGFX0019, d.loc, "A __shared__ variable cannot be initialized at its declaration");
                analyzeExpr(d.init);
            }
        } else {
            sym.kind = SymbolKind::Local;
            sym.emitName = safeIdentifier(d.name);
        }
        d.emitName = sym.emitName;

        if (d.init && !d.isShared) {
            AtomicOp op;
            if (d.init->kind == ExprKind::Call && parseAtomicName(static_cast<CallExpr*>(d.init.get())->callee, op)) {
                d.init = buildAtomic(*static_cast<CallExpr*>(d.init.get()), op, d.emitName, bad ? nullptr : &d.type);
            } else {
                analyzeExpr(d.init);
                if (!bad) {
                    if (d.type.isArray()) {
                        diags_.reportError(DiagCode::CGFX0009, d.loc, "Array initialization is not supported: initialize element by element");
                    } else {
                        coerce(d.init, d.type, d.loc, "initialization");
                    }
                }
            }
        }
        declared_.insert(d.name);
        declare(d.name, sym, d.loc);
    }
}

// ---------------------------------------------------------------------------------------------
// Call graph: recursion (CGFX0005) and per-kernel closure
// ---------------------------------------------------------------------------------------------

void Sema::buildCallGraph() {
    std::unordered_map<const FunctionDecl*, int> color;
    std::vector<const FunctionDecl*> stack;
    std::function<void(const FunctionDecl*)> dfs = [&](const FunctionDecl* f) {
        color[f] = 1;
        stack.push_back(f);
        const auto it = callSites_.find(f);
        if (it != callSites_.end()) {
            for (const auto& cs : it->second) {
                const int c = color[cs.callee];
                if (c == 1) {
                    std::string path;
                    bool inCycle = false;
                    for (const FunctionDecl* s : stack) {
                        if (s == cs.callee) inCycle = true;
                        if (inCycle) path += s->name + " -> ";
                    }
                    path += cs.callee->name;
                    diags_.reportError(DiagCode::CGFX0005, cs.loc,
                                       "Recursion detected (" + path + "): not supported in device code");
                } else if (c == 0) {
                    dfs(cs.callee);
                }
            }
        }
        stack.pop_back();
        color[f] = 2;
    };
    for (auto& f : module_->functions) {
        if (f->body && color[f.get()] == 0) dfs(f.get());
    }

    for (auto& k : module_->functions) {
        if (!k->isKernel || !k->body) continue;
        std::vector<const FunctionDecl*> order;
        std::unordered_set<const FunctionDecl*> visited;
        std::function<void(const FunctionDecl*)> post = [&](const FunctionDecl* f) {
            const auto it = callSites_.find(f);
            if (it != callSites_.end()) {
                for (const auto& cs : it->second) {
                    if (visited.insert(cs.callee).second) {
                        post(cs.callee);
                        order.push_back(cs.callee);
                    }
                }
            }
        };
        visited.insert(k.get());
        post(k.get());
        for (const FunctionDecl* h : order) {
            if (h->usesGridDim) k->usesGridDim = true;
            if (h->usesPopcount) k->usesPopcount = true;
        }
        module_->helperClosure[k.get()] = std::move(order);
    }
}

void Sema::validateLaunchSites() {
    for (const auto& site : module_->launchSites) {
        const auto it = functions_.find(site.kernelName);
        if (it == functions_.end()) {
            diags_.reportError(DiagCode::CGFX0010, site.kernelLoc,
                               "Unknown kernel: '" + site.kernelName + "' (no visible __global__ declaration)");
            continue;
        }
        const FunctionDecl& f = *it->second;
        if (!f.isKernel) {
            diags_.reportError(DiagCode::CGFX0019, site.kernelLoc, "'" + site.kernelName + "' is not a kernel (__global__)");
            continue;
        }
        if (!f.body) {
            diags_.reportError(DiagCode::CGFX0010, site.kernelLoc,
                               "Kernel '" + site.kernelName + "' is declared but its definition is not visible");
            continue;
        }
        const unsigned x = site.evaluatedBlock[0], y = site.evaluatedBlock[1], z = site.evaluatedBlock[2];
        const unsigned long long threads = static_cast<unsigned long long>(x) * y * z;
        if (x > kMaxBlockX || y > kMaxBlockY || z > kMaxBlockZ || threads > kMaxThreadsPerBlock) {
            diags_.reportError(DiagCode::CGFX0008, site.kernelLoc,
                               "Block size " + std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z) +
                                   " is beyond the limits (x <= 1024, y <= 1024, z <= 64, at most 1024 threads in total)");
        }
        if (f.launchBounds != 0 && threads != f.launchBounds) {
            diags_.reportWarning(DiagCode::CGFX0007, site.kernelLoc,
                                 "Block size " + std::to_string(threads) + " is inconsistent with __launch_bounds__(" +
                                     std::to_string(f.launchBounds) + ") of kernel '" + f.name + "'");
        }
        if (site.argExprTexts.size() != f.parameters.size()) {
            diags_.reportError(DiagCode::CGFX0019, site.kernelLoc,
                               "Kernel '" + f.name + "' expects " + std::to_string(f.parameters.size()) +
                                   " argument(s), " + std::to_string(site.argExprTexts.size()) + " provided at the launch");
        }
    }
}

void Sema::checkNameCollisions() {
    for (auto& k : module_->functions) {
        if (!k->isKernel || !k->body) continue;
        bool anyBuffer = false;
        for (const auto& p : k->parameters) anyBuffer = anyBuffer || p.type.isPointer();
        if (!anyBuffer) continue;

        // Names declared by the kernel and its functions: a small local set; the global names are
        // looked up in place (never copied: the cost would be quadratic on a large file).
        std::vector<const std::unordered_set<std::string>*> scopes;
        const auto self = declaredNames_.find(k.get());
        if (self != declaredNames_.end()) scopes.push_back(&self->second);
        const auto clo = module_->helperClosure.find(k.get());
        if (clo != module_->helperClosure.end()) {
            for (const FunctionDecl* h : clo->second) {
                const auto d = declaredNames_.find(h);
                if (d != declaredNames_.end()) scopes.push_back(&d->second);
            }
        }
        for (const auto& p : k->parameters) {
            if (!p.type.isPointer()) continue;
            const std::string block = p.emitName + "Buffer";
            bool clash = globalNames_.count(block) != 0;
            for (const auto* set : scopes) clash = clash || set->count(block) != 0;
            if (clash) {
                diags_.reportError(DiagCode::CGFX0016, p.loc,
                                   "Name collision: '" + block + "' is the name of the buffer block generated for parameter '" +
                                       p.name + "' (rename one of the two)");
            }
        }
    }
}

} // namespace cudagfx

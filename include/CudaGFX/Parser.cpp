// Parser.cpp: declarations, types, statements and expressions of the device code.
#include "Parser.hpp"

#include "Literal.hpp"
#include "ShaderNames.hpp"

namespace cudagfx {

namespace {

bool binaryOpFor(TokenType t, BinaryOp& op, int& prec, bool& rightAssoc) {
    rightAssoc = false;
    switch (t) {
        case TokenType::Assign: op = BinaryOp::Assign; prec = 1; rightAssoc = true; return true;
        case TokenType::PlusAssign: op = BinaryOp::AddAssign; prec = 1; rightAssoc = true; return true;
        case TokenType::MinusAssign: op = BinaryOp::SubAssign; prec = 1; rightAssoc = true; return true;
        case TokenType::StarAssign: op = BinaryOp::MulAssign; prec = 1; rightAssoc = true; return true;
        case TokenType::SlashAssign: op = BinaryOp::DivAssign; prec = 1; rightAssoc = true; return true;
        case TokenType::PercentAssign: op = BinaryOp::ModAssign; prec = 1; rightAssoc = true; return true;
        case TokenType::LessLessAssign: op = BinaryOp::ShlAssign; prec = 1; rightAssoc = true; return true;
        case TokenType::GreaterGreaterAssign: op = BinaryOp::ShrAssign; prec = 1; rightAssoc = true; return true;
        case TokenType::AmpAssign: op = BinaryOp::AndAssign; prec = 1; rightAssoc = true; return true;
        case TokenType::PipeAssign: op = BinaryOp::OrAssign; prec = 1; rightAssoc = true; return true;
        case TokenType::CaretAssign: op = BinaryOp::XorAssign; prec = 1; rightAssoc = true; return true;
        case TokenType::PipePipe: op = BinaryOp::LogOr; prec = 3; return true;
        case TokenType::AmpAmp: op = BinaryOp::LogAnd; prec = 4; return true;
        case TokenType::Pipe: op = BinaryOp::BitOr; prec = 5; return true;
        case TokenType::Caret: op = BinaryOp::BitXor; prec = 6; return true;
        case TokenType::Amp: op = BinaryOp::BitAnd; prec = 7; return true;
        case TokenType::Equal: op = BinaryOp::Eq; prec = 8; return true;
        case TokenType::NotEqual: op = BinaryOp::Ne; prec = 8; return true;
        case TokenType::Less: op = BinaryOp::Lt; prec = 9; return true;
        case TokenType::LessEqual: op = BinaryOp::Le; prec = 9; return true;
        case TokenType::Greater: op = BinaryOp::Gt; prec = 9; return true;
        case TokenType::GreaterEqual: op = BinaryOp::Ge; prec = 9; return true;
        case TokenType::LessLess: op = BinaryOp::Shl; prec = 10; return true;
        case TokenType::GreaterGreater: op = BinaryOp::Shr; prec = 10; return true;
        case TokenType::Plus: op = BinaryOp::Add; prec = 11; return true;
        case TokenType::Minus: op = BinaryOp::Sub; prec = 11; return true;
        case TokenType::Star: op = BinaryOp::Mul; prec = 12; return true;
        case TokenType::Slash: op = BinaryOp::Div; prec = 12; return true;
        case TokenType::Percent: op = BinaryOp::Mod; prec = 12; return true;
        default: return false;
    }
}

bool isBuiltinName(const std::string& n, BuiltinExpr::Kind& kind) {
    if (n == "threadIdx") { kind = BuiltinExpr::Kind::ThreadIdx; return true; }
    if (n == "blockIdx") { kind = BuiltinExpr::Kind::BlockIdx; return true; }
    if (n == "blockDim") { kind = BuiltinExpr::Kind::BlockDim; return true; }
    if (n == "gridDim") { kind = BuiltinExpr::Kind::GridDim; return true; }
    return false;
}

Type vectorTypeFromName(const std::string& name) {
    if (name.size() < 4) return Type::makeUnknown();
    const unsigned n = static_cast<unsigned>(name.back() - '0');
    if (name.compare(0, 4, "uint") == 0) return Type::makeVector(Type::Kind::UInt, n);
    if (name.compare(0, 3, "int") == 0) return Type::makeVector(Type::Kind::Int, n);
    if (name.compare(0, 5, "float") == 0) return Type::makeVector(Type::Kind::Float, n);
    return Type::makeUnknown();
}

} // namespace

Parser::Parser(const Preprocessor& pp, const SourceManager& sources, std::vector<Token> activeTokens,
               FileId mainFile, DiagnosticsEngine& diags)
    : pp_(pp), sources_(sources), raw_(std::move(activeTokens)), mainFile_(mainFile), diags_(diags) {
    toks_ = pp_.expand(raw_);
    if (toks_.empty() || toks_.back().type != TokenType::EndOfFile) {
        Token eof;
        eof.type = TokenType::EndOfFile;
        if (!toks_.empty()) eof.location = toks_.back().location;
        toks_.push_back(eof);
    }
    module_.mainFile = mainFile;
}

// ---------------------------------------------------------------------------------------------
// Cursor and errors
// ---------------------------------------------------------------------------------------------

const Token& Parser::peek(std::size_t ahead) const {
    const std::size_t i = pos_ + ahead;
    return i < toks_.size() ? toks_[i] : toks_.back();
}

const Token& Parser::get() {
    if (pos_ + 1 < toks_.size()) return toks_[pos_++];
    return toks_.back();
}

bool Parser::match(TokenType type) {
    if (!check(type)) return false;
    get();
    return true;
}

bool Parser::expect(TokenType type, const char* what) {
    if (check(type)) {
        get();
        return true;
    }
    const Token& t = peek();
    errorAt(t.location, std::string("Expected: ") + what + (t.type == TokenType::EndOfFile
                                                              ? " (end of file reached)"
                                                              : ", found '" + t.text + "'"));
    return false;
}

std::size_t Parser::findMatching(std::size_t openIndex, TokenType open, TokenType close) const {
    int depth = 0;
    for (std::size_t i = openIndex; i < toks_.size(); ++i) {
        if (toks_[i].type == open) ++depth;
        else if (toks_[i].type == close && --depth == 0) return i;
        else if (toks_[i].type == TokenType::EndOfFile) break;
    }
    return std::string::npos;
}

void Parser::errorAt(const SourceLocation& loc, const std::string& message) {
    if (!panic_) diags_.reportError(DiagCode::CGFX0018, loc, message);
    panic_ = true;
}

void Parser::unsupported(const SourceLocation& loc, const std::string& what) {
    diags_.reportError(DiagCode::CGFX0009, loc, what);
}

void Parser::synchronize() {
    while (!atEnd()) {
        if (check(TokenType::Semi)) {
            get();
            break;
        }
        if (check(TokenType::RBrace)) break;
        get();
    }
    panic_ = false;
}

void Parser::skipBalanced(TokenType open, TokenType close) {
    const std::size_t end = findMatching(pos_, open, close);
    if (end == std::string::npos) {
        pos_ = toks_.size() - 1;
    } else {
        pos_ = end + 1;
    }
}

// ---------------------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------------------

bool Parser::resolveConst(const std::string& name, std::int64_t& value) const {
    for (auto it = localConsts_.rbegin(); it != localConsts_.rend(); ++it) {
        const auto f = it->find(name);
        if (f != it->end()) {
            value = f->second;
            return true;
        }
    }
    const auto g = globalInts_.find(name);
    if (g != globalInts_.end()) {
        value = g->second;
        return true;
    }
    return false;
}

bool Parser::evalConst(std::size_t first, std::size_t last, std::int64_t& out) {
    if (first >= last || last > toks_.size()) return false;
    ConstEvaluator ev([this](const std::string& n, std::int64_t& v) { return resolveConst(n, v); });
    return ev.evaluate(toks_.data() + first, toks_.data() + last, out);
}

// ---------------------------------------------------------------------------------------------
// File level
// ---------------------------------------------------------------------------------------------

ParsedModule Parser::parse() {
    scanLaunchSites();
    parseTopLevel();
    return std::move(module_);
}

bool Parser::isTransparentBrace(std::size_t braceIndex) const {
    // namespace [a::b] {   or   extern "C" {
    if (braceIndex == 0) return false;
    std::size_t j = braceIndex - 1;
    if (toks_[j].type == TokenType::KwNamespace) return true;
    if (toks_[j].type == TokenType::StringLiteral && j >= 1 && toks_[j - 1].type == TokenType::KwExtern) return true;
    while (toks_[j].type == TokenType::Identifier) {
        if (j >= 1 && toks_[j - 1].type == TokenType::KwNamespace) return true;
        if (j >= 2 && toks_[j - 1].type == TokenType::ColonColon) {
            j -= 2;
            continue;
        }
        break;
    }
    return false;
}

void Parser::parseTopLevel() {
    std::vector<bool> braces; // true = transparent block (namespace, extern "C")
    std::size_t declStart = 0;
    pos_ = 0;
    while (!atEnd()) {
        const TokenType k = peek().type;
        if (k == TokenType::LBrace) {
            const bool transparent = isTransparentBrace(pos_);
            braces.push_back(transparent);
            if (!transparent) ++opaqueDepth_;
            get();
            if (transparent) declStart = pos_;
            continue;
        }
        if (k == TokenType::RBrace) {
            if (!braces.empty()) {
                if (!braces.back()) --opaqueDepth_;
                braces.pop_back();
            }
            get();
            if (opaqueDepth_ == 0) declStart = pos_;
            continue;
        }
        if (k == TokenType::Semi) {
            get();
            if (opaqueDepth_ == 0) declStart = pos_;
            continue;
        }
        if (opaqueDepth_ == 0) {
            if (k == TokenType::KwStruct && pos_ == declStart) {
                if (tryParseStruct()) {
                    declStart = pos_;
                    continue;
                }
            } else if ((k == TokenType::KwConst || k == TokenType::KwConstexpr || k == TokenType::KwStatic ||
                        k == TokenType::KwInline) && pos_ == declStart) {
                if (tryParseGlobalConst()) {
                    declStart = pos_;
                    continue;
                }
            }
            if (k == TokenType::KwGlobal || k == TokenType::KwDevice || k == TokenType::KwConstant) {
                parseDeviceDeclaration(declStart);
                declStart = pos_;
                continue;
            }
        }
        get();
    }
}

void Parser::parseDeviceDeclaration(std::size_t declStart) {
    auto fn = parseFunction(declStart);
    panic_ = false;
    if (fn) module_.functions.push_back(std::move(fn));
}

unsigned Parser::parseLaunchBounds() {
    get(); // __launch_bounds__
    unsigned result = 0;
    if (!expect(TokenType::LParen, "'(' after __launch_bounds__")) return 0;
    const std::size_t close = findMatching(pos_ - 1, TokenType::LParen, TokenType::RParen);
    if (close == std::string::npos) {
        errorAt(peek().location, "unterminated __launch_bounds__");
        return 0;
    }
    // First argument: up to the first comma at depth 0.
    std::size_t end = pos_;
    int depth = 0;
    while (end < close) {
        if (toks_[end].type == TokenType::LParen) ++depth;
        else if (toks_[end].type == TokenType::RParen) --depth;
        else if (toks_[end].type == TokenType::Comma && depth == 0) break;
        ++end;
    }
    std::int64_t v = 0;
    if (evalConst(pos_, end, v) && v > 0 && v <= 65536) {
        result = static_cast<unsigned>(v);
    } else {
        diags_.reportError(DiagCode::CGFX0006, peek().location,
                           "The argument of __launch_bounds__ must be a strictly positive integer constant");
    }
    pos_ = close + 1;
    return result;
}

std::unique_ptr<FunctionDecl> Parser::parseFunction(std::size_t declStart) {
    auto fn = std::make_unique<FunctionDecl>();
    const SourceLocation startLoc = toks_[declStart].location;
    fn->loc = startLoc;
    const std::size_t prefixEnd = pos_;
    pos_ = declStart;

    // `template<...>` header: rejected, but we still parse the function (with diagnostics muted)
    // to know where it ends, then discard it.
    bool isTemplate = false;
    SourceLocation templateLoc;
    for (std::size_t i = declStart; i < prefixEnd; ++i) {
        if (toks_[i].type == TokenType::KwTemplate) {
            isTemplate = true;
            templateLoc = toks_[i].location;
            break;
        }
    }
    if (isTemplate) {
        unsupported(templateLoc, "Templates are not supported (kernels and __device__ functions must be non-generic)");
        diags_.setMuted(true);
    }
    struct MuteGuard {
        DiagnosticsEngine& d;
        bool active;
        ~MuteGuard() { if (active) d.setMuted(false); }
    } guard{diags_, isTemplate};

    // Qualifiers
    while (true) {
        const TokenType k = peek().type;
        if (k == TokenType::KwGlobal) { fn->isKernel = true; get(); }
        else if (k == TokenType::KwDevice) { fn->isDevice = true; get(); }
        else if (k == TokenType::KwHost) { fn->isHost = true; get(); }
        else if (k == TokenType::KwConstant) { get(); fn->isDevice = true; }
        else if (k == TokenType::KwStatic || k == TokenType::KwInline || k == TokenType::KwConstexpr ||
                 k == TokenType::KwVolatile) { get(); }
        else if (k == TokenType::KwExtern) {
            get();
            if (check(TokenType::StringLiteral)) get();
        }
        else if (k == TokenType::KwTemplate) {
            get();
            if (check(TokenType::Less)) {
                int depth = 0;
                while (!atEnd()) {
                    if (check(TokenType::Less)) ++depth;
                    else if (check(TokenType::Greater) && --depth == 0) { get(); break; }
                    else if (check(TokenType::GreaterGreater)) { depth -= 2; if (depth <= 0) { get(); break; } }
                    get();
                }
            }
        }
        else if (k == TokenType::KwLaunchBounds) { fn->launchBounds = parseLaunchBounds(); }
        else if (k == TokenType::Identifier && peek().text == "__attribute__") {
            get();
            if (check(TokenType::LParen)) skipBalanced(TokenType::LParen, TokenType::RParen);
        }
        else break;
    }

    // A global __device__/__constant__ variable is not a function.
    fn->returnType = parseType();
    if (!check(TokenType::Identifier)) {
        errorAt(peek().location, "Function name expected");
        synchronize();
        return nullptr;
    }
    const Token nameTok = get();
    fn->name = nameTok.text;
    fn->loc = nameTok.location;
    if (!check(TokenType::LParen)) {
        unsupported(nameTok.location,
                    "Global __device__/__constant__ variable not supported ('" + fn->name +
                        "'): pass the value as a kernel parameter or use const/constexpr");
        while (!atEnd() && !check(TokenType::Semi)) {
            if (check(TokenType::LBrace)) skipBalanced(TokenType::LBrace, TokenType::RBrace);
            else get();
        }
        match(TokenType::Semi);
        return nullptr;
    }
    get(); // '('

    if (check(TokenType::KwVoid) && peek(1).type == TokenType::RParen) {
        get();
    } else if (!check(TokenType::RParen)) {
        while (!atEnd()) {
            Parameter p;
            if (!parseParameter(p)) break;
            fn->parameters.push_back(std::move(p));
            if (!match(TokenType::Comma)) break;
        }
    }
    if (!expect(TokenType::RParen, "')' after the parameters")) {
        synchronize();
        return nullptr;
    }
    while (check(TokenType::KwConst) || (check(TokenType::Identifier) && (peek().text == "noexcept" || peek().text == "override"))) get();

    if (check(TokenType::Semi)) {
        const Token semi = get();
        fn->span.start = startLoc;
        fn->span.end = semi.location;
        fn->span.end.offset = semi.endOffset();
        return isTemplate ? nullptr : std::move(fn);
    }
    if (!check(TokenType::LBrace)) {
        errorAt(peek().location, "Function body '{' or ';' expected");
        synchronize();
        return nullptr;
    }
    fn->body = parseCompound();
    fn->span.start = startLoc;
    fn->span.end = fn->body->endLoc;
    fn->span.end.offset += 1; // the closing brace takes one byte
    return isTemplate ? nullptr : std::move(fn);
}

bool Parser::parseParameter(Parameter& out) {
    if (!isTypeKeyword(peek().type) && !check(TokenType::Identifier) && !check(TokenType::KwStruct)) {
        errorAt(peek().location, "Parameter type expected, found '" + peek().text + "'");
        return false;
    }
    out.type = parseType();
    out.loc = peek().location;
    if (check(TokenType::Identifier)) {
        out.name = peek().text;
        out.loc = peek().location;
        get();
    }
    // `T x[]` or `T x[N]`: an array parameter is a pointer.
    while (check(TokenType::LBracket)) {
        skipBalanced(TokenType::LBracket, TokenType::RBracket);
        out.type = Type::makePointer(out.type, out.type.isConst);
    }
    if (check(TokenType::Assign)) {
        unsupported(peek().location, "Default parameter values are not supported");
        get();
        parseExpression(1);
    }
    return true;
}

bool Parser::tryParseStruct() {
    const std::size_t save = pos_;
    const bool savedPanic = panic_;
    diags_.setMuted(true);
    diags_.resetMutedErrors();

    bool ok = true;
    auto decl = std::make_unique<StructDecl>();
    get(); // struct
    if (!check(TokenType::Identifier)) ok = false;
    if (ok) {
        decl->name = peek().text;
        decl->loc = peek().location;
        get();
        if (!check(TokenType::LBrace)) ok = false;
    }
    if (ok) {
        get(); // {
        while (ok && !check(TokenType::RBrace) && !atEnd()) {
            if (!(isTypeKeyword(peek().type) || check(TokenType::Identifier) || check(TokenType::KwStruct))) { ok = false; break; }
            Type base = parseType(false);
            do {
                Type t = applyPointers(base);
                if (!check(TokenType::Identifier)) { ok = false; break; }
                StructField f;
                f.name = peek().text;
                get();
                std::vector<unsigned> dims;
                while (ok && check(TokenType::LBracket)) {
                    const std::size_t close = findMatching(pos_, TokenType::LBracket, TokenType::RBracket);
                    std::int64_t v = 0;
                    if (close == std::string::npos || !evalConst(pos_ + 1, close, v) || v <= 0 || v > (1 << 20)) { ok = false; break; }
                    dims.push_back(static_cast<unsigned>(v));
                    pos_ = close + 1;
                }
                for (auto it = dims.rbegin(); it != dims.rend(); ++it) t = Type::makeArray(t, *it);
                f.type = t;
                f.emitName = safeIdentifier(f.name);
                decl->fields.push_back(std::move(f));
            } while (ok && match(TokenType::Comma));
            if (ok && !match(TokenType::Semi)) ok = false;
        }
        if (ok && !match(TokenType::RBrace)) ok = false;
        if (ok && !match(TokenType::Semi)) ok = false; // `struct S {...} s;`: a declarator, not a plain POD
    }
    if (diags_.mutedErrorCount() > 0) ok = false;
    diags_.setMuted(false);
    diags_.resetMutedErrors();
    panic_ = savedPanic;
    if (!ok || decl->fields.empty()) {
        pos_ = save;
        return false;
    }
    decl->emitName = safeIdentifier(decl->name);
    structNames_.insert(decl->name);
    module_.structs.push_back(std::move(decl));
    return true;
}

bool Parser::tryParseGlobalConst() {
    const std::size_t save = pos_;
    const bool savedPanic = panic_;
    diags_.setMuted(true);
    diags_.resetMutedErrors();

    bool ok = true;
    bool isConst = false;
    while (check(TokenType::KwStatic) || check(TokenType::KwInline) || check(TokenType::KwConst) || check(TokenType::KwConstexpr)) {
        if (check(TokenType::KwConst) || check(TokenType::KwConstexpr)) isConst = true;
        get();
    }
    auto decl = std::make_unique<GlobalConstDecl>();
    decl->declStart = toks_[save].location;
    std::size_t initFirst = 0, initLast = 0;
    if (!isConst || !isTypeKeyword(peek().type)) ok = false;
    if (ok) {
        decl->type = parseType(false);
        decl->type.isConst = true;
        if (!(decl->type.isScalar() || decl->type.isVector()) || !check(TokenType::Identifier)) ok = false;
    }
    if (ok) {
        decl->name = peek().text;
        decl->loc = peek().location;
        get();
        if (!check(TokenType::Assign)) ok = false;
    }
    if (ok) {
        get();
        initFirst = pos_;
        decl->init = parseExpression(1);
        initLast = pos_;
        if (!check(TokenType::Semi)) ok = false;
    }
    if (diags_.mutedErrorCount() > 0) ok = false;
    diags_.setMuted(false);
    diags_.resetMutedErrors();
    panic_ = savedPanic;
    if (!ok) {
        pos_ = save;
        return false;
    }
    get(); // ';'
    std::int64_t v = 0;
    if ((decl->type.isInteger() || decl->type.kind == Type::Kind::Bool) && evalConst(initFirst, initLast, v)) {
        globalInts_[decl->name] = v;
    }
    module_.globals.push_back(std::move(decl));
    return true;
}

// ---------------------------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------------------------

bool Parser::isTypeKeyword(TokenType t) const {
    switch (t) {
        case TokenType::KwVoid: case TokenType::KwBool: case TokenType::KwChar: case TokenType::KwShort:
        case TokenType::KwInt: case TokenType::KwUnsigned: case TokenType::KwSigned: case TokenType::KwLong:
        case TokenType::KwFloat: case TokenType::KwDouble: case TokenType::KwUint: case TokenType::KwVecType:
        case TokenType::KwConst: case TokenType::KwConstexpr: case TokenType::KwStatic: case TokenType::KwVolatile:
        case TokenType::KwInline: case TokenType::KwAuto:
            return true;
        default:
            return false;
    }
}

bool Parser::isCastStart() const {
    if (!check(TokenType::LParen)) return false;
    switch (peek(1).type) {
        case TokenType::KwVoid: case TokenType::KwBool: case TokenType::KwChar: case TokenType::KwShort:
        case TokenType::KwInt: case TokenType::KwUnsigned: case TokenType::KwSigned: case TokenType::KwLong:
        case TokenType::KwFloat: case TokenType::KwDouble: case TokenType::KwUint: case TokenType::KwVecType:
        case TokenType::KwConst:
            return true;
        default:
            return false;
    }
}

bool Parser::isDeclStart() const {
    switch (peek().type) {
        case TokenType::KwConst: case TokenType::KwConstexpr: case TokenType::KwStatic: case TokenType::KwVolatile:
        case TokenType::KwShared: case TokenType::KwExtern: case TokenType::KwStruct:
            return true;
        case TokenType::KwBool: case TokenType::KwChar: case TokenType::KwShort: case TokenType::KwInt:
        case TokenType::KwUnsigned: case TokenType::KwSigned: case TokenType::KwLong: case TokenType::KwFloat:
        case TokenType::KwDouble: case TokenType::KwUint: case TokenType::KwVecType:
            return peek(1).type != TokenType::LParen; // otherwise: functional cast float(x)
        case TokenType::Identifier:
            return peek(1).type == TokenType::Identifier;
        default:
            return false;
    }
}

Type Parser::parseBaseType() {
    const Token t = peek();
    switch (t.type) {
        case TokenType::KwVoid: get(); return Type::makeVoid();
        case TokenType::KwBool: get(); return Type::makeBool();
        case TokenType::KwInt: get(); return Type::makeInt();
        case TokenType::KwUint: get(); return Type::makeUInt();
        case TokenType::KwFloat: get(); return Type::makeFloat();
        case TokenType::KwDouble: get(); return Type::makeUnsupported("double");
        case TokenType::KwVecType: get(); return vectorTypeFromName(t.text);
        case TokenType::KwChar: get(); return Type::makeUnsupported("char");
        case TokenType::KwShort:
            get();
            match(TokenType::KwInt);
            return Type::makeUnsupported("short");
        case TokenType::KwUnsigned:
        case TokenType::KwSigned: {
            get();
            const bool isUnsigned = t.type == TokenType::KwUnsigned;
            if (match(TokenType::KwInt)) return isUnsigned ? Type::makeUInt() : Type::makeInt();
            if (match(TokenType::KwChar)) return Type::makeUnsupported(isUnsigned ? "unsigned char" : "signed char");
            if (match(TokenType::KwShort)) { match(TokenType::KwInt); return Type::makeUnsupported(isUnsigned ? "unsigned short" : "short"); }
            if (match(TokenType::KwLong)) {
                match(TokenType::KwLong);
                match(TokenType::KwInt);
                return Type::makeUnsupported(isUnsigned ? "unsigned long" : "long");
            }
            return isUnsigned ? Type::makeUInt() : Type::makeInt();
        }
        case TokenType::KwLong: {
            get();
            std::string name = "long";
            if (match(TokenType::KwLong)) name = "long long";
            else if (match(TokenType::KwDouble)) name = "long double";
            match(TokenType::KwInt);
            return Type::makeUnsupported(name);
        }
        case TokenType::KwStruct: {
            get();
            if (check(TokenType::Identifier)) {
                const std::string n = get().text;
                return Type::makeStruct(safeIdentifier(n));
            }
            errorAt(peek().location, "Struct name expected");
            return Type::makeUnknown();
        }
        case TokenType::KwAuto:
            unsupported(t.location, "Type deduction (auto) is not supported: write the type explicitly");
            get();
            return Type::makeUnknown();
        case TokenType::Identifier: {
            std::string name = get().text;
            while (check(TokenType::ColonColon) && peek(1).type == TokenType::Identifier) {
                get();
                name += "::" + get().text;
            }
            if (structNames_.count(name)) return Type::makeStruct(safeIdentifier(name));
            return Type::makeUnsupported(name);
        }
        default:
            errorAt(t.location, "Type expected, found '" + t.text + "'");
            return Type::makeUnknown();
    }
}

Type Parser::applyPointers(Type base) {
    Type t = std::move(base);
    while (true) {
        if (check(TokenType::Star)) {
            get();
            t = Type::makePointer(t, t.isConst);
            while (check(TokenType::KwConst) || check(TokenType::KwRestrict) || check(TokenType::KwVolatile)) get();
        } else if (check(TokenType::Amp)) {
            unsupported(peek().location, "C++ reference (&) is not supported: pass by value");
            get();
        } else {
            break;
        }
    }
    return t;
}

Type Parser::parseType(bool allowPointers) {
    bool isConst = false;
    while (check(TokenType::KwConst) || check(TokenType::KwConstexpr) || check(TokenType::KwVolatile) ||
           check(TokenType::KwStatic) || check(TokenType::KwInline) || check(TokenType::KwRestrict)) {
        if (check(TokenType::KwConst) || check(TokenType::KwConstexpr)) isConst = true;
        get();
    }
    Type base = parseBaseType();
    while (check(TokenType::KwConst) || check(TokenType::KwVolatile)) {
        if (check(TokenType::KwConst)) isConst = true;
        get();
    }
    if (isConst) base.isConst = true;
    return allowPointers ? applyPointers(std::move(base)) : base;
}

// ---------------------------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------------------------

std::unique_ptr<CompoundStmt> Parser::parseCompound() {
    auto block = std::make_unique<CompoundStmt>();
    block->loc = peek().location;
    if (!expect(TokenType::LBrace, "'{'")) {
        block->endLoc = block->loc;
        return block;
    }
    localConsts_.emplace_back();
    while (!check(TokenType::RBrace) && !atEnd()) {
        const std::size_t before = pos_;
        StmtPtr s = parseStatement();
        if (s) block->statements.push_back(std::move(s));
        if (panic_) synchronize();
        if (pos_ == before) get(); // guarantees progress
    }
    block->endLoc = peek().location;
    localConsts_.pop_back();
    if (!match(TokenType::RBrace)) errorAt(peek().location, "'}' expected (end of file reached)");
    return block;
}

StmtPtr Parser::parseStatement() {
    const Token& t = peek();
    switch (t.type) {
        case TokenType::LBrace: return parseCompound();
        case TokenType::KwIf: return parseIf();
        case TokenType::KwFor: return parseFor();
        case TokenType::KwWhile: return parseWhile();
        case TokenType::KwDo: return parseDoWhile();
        case TokenType::KwSwitch: return parseSwitch();
        case TokenType::KwReturn: return parseReturn();
        case TokenType::KwBreak: {
            auto s = std::make_unique<BreakStmt>();
            s->loc = get().location;
            expect(TokenType::Semi, "';'");
            return s;
        }
        case TokenType::KwContinue: {
            auto s = std::make_unique<ContinueStmt>();
            s->loc = get().location;
            expect(TokenType::Semi, "';'");
            return s;
        }
        case TokenType::Semi: {
            auto s = std::make_unique<EmptyStmt>();
            s->loc = get().location;
            return s;
        }
        case TokenType::KwSyncthreads: {
            auto s = std::make_unique<SyncThreadsStmt>();
            s->loc = get().location;
            expect(TokenType::LParen, "'(' after __syncthreads");
            expect(TokenType::RParen, "')' after __syncthreads(");
            expect(TokenType::Semi, "';'");
            return s;
        }
        case TokenType::KwCase:
        case TokenType::KwDefault:
            errorAt(t.location, "'" + t.text + "' outside of a switch");
            return nullptr;
        default:
            break;
    }
    if (isUnsupportedKeyword(t.type)) {
        unsupported(t.location, "C++ construct not supported in device code: '" + t.text + "'");
        synchronize();
        return nullptr;
    }
    if (isDeclStart()) return parseDeclStmt();

    auto s = std::make_unique<ExprStmt>();
    s->loc = t.location;
    s->expr = parseCommaExpression();
    expect(TokenType::Semi, "';'");
    return s;
}

StmtPtr Parser::parseDeclStmt(bool consumeSemicolon) {
    auto stmt = std::make_unique<DeclStmt>();
    stmt->loc = peek().location;
    bool isShared = false, isConst = false, isExtern = false;
    while (true) {
        if (match(TokenType::KwShared)) isShared = true;
        else if (check(TokenType::KwExtern)) { get(); isExtern = true; }
        else if (check(TokenType::KwConst) || check(TokenType::KwConstexpr)) { isConst = true; get(); }
        else if (check(TokenType::KwStatic) || check(TokenType::KwVolatile)) get();
        else break;
    }
    if (isExtern && isShared) {
        diags_.reportError(DiagCode::CGFX0004, stmt->loc,
                           "extern __shared__ (dynamic shared memory) is not supported: declare a statically sized array");
        synchronize();
        return std::make_unique<EmptyStmt>();
    }
    const Type base = parseType(false);

    do {
        Type t = applyPointers(base);
        if (!check(TokenType::Identifier)) {
            errorAt(peek().location, "Variable name expected, found '" + peek().text + "'");
            break;
        }
        const Token nameTok = get();
        std::vector<unsigned> dims;
        while (check(TokenType::LBracket)) {
            const std::size_t close = findMatching(pos_, TokenType::LBracket, TokenType::RBracket);
            if (close == std::string::npos) {
                errorAt(peek().location, "']' expected");
                break;
            }
            std::int64_t v = 0;
            if (close == pos_ + 1) {
                if (isShared) {
                    diags_.reportError(DiagCode::CGFX0004, peek().location,
                                       "__shared__ array without a size: the size must be static");
                } else {
                    errorAt(peek().location, "Array size required");
                }
                dims.push_back(1);
            } else if (!evalConst(pos_ + 1, close, v) || v <= 0 || v > (1 << 24)) {
                if (isShared) {
                    diags_.reportError(DiagCode::CGFX0004, toks_[pos_ + 1].location,
                                       "The size of a __shared__ array must be a positive integer constant");
                } else {
                    diags_.reportError(DiagCode::CGFX0019, toks_[pos_ + 1].location,
                                       "The size of an array must be a positive integer constant");
                }
                dims.push_back(1);
            } else {
                dims.push_back(static_cast<unsigned>(v));
            }
            pos_ = close + 1;
        }
        for (auto it = dims.rbegin(); it != dims.rend(); ++it) t = Type::makeArray(t, *it);

        DeclStmt::SingleDecl d;
        d.type = t;
        d.name = nameTok.text;
        d.loc = nameTok.location;
        d.isShared = isShared;
        d.isConst = isConst || base.isConst;
        std::size_t initFirst = 0, initLast = 0;
        if (match(TokenType::Assign)) {
            if (check(TokenType::LBrace)) {
                unsupported(peek().location, "Initializer list {...} is not supported: initialize element by element");
                skipBalanced(TokenType::LBrace, TokenType::RBrace);
            } else {
                initFirst = pos_;
                d.init = parseExpression(1);
                initLast = pos_;
            }
        } else if (check(TokenType::LParen) || check(TokenType::LBrace)) {
            unsupported(peek().location, "Constructor-style initialization is not supported: use '= value'");
            if (check(TokenType::LParen)) skipBalanced(TokenType::LParen, TokenType::RParen);
            else skipBalanced(TokenType::LBrace, TokenType::RBrace);
        }
        if (d.isConst && (t.isInteger() || t.kind == Type::Kind::Bool) && d.init) {
            std::int64_t v = 0;
            if (evalConst(initFirst, initLast, v)) localConsts_.back()[d.name] = v;
        }
        stmt->decls.push_back(std::move(d));
        if (panic_) return stmt;
    } while (match(TokenType::Comma));

    if (consumeSemicolon) expect(TokenType::Semi, "';'");
    return stmt;
}

StmtPtr Parser::parseIf() {
    auto s = std::make_unique<IfStmt>();
    s->loc = get().location;
    expect(TokenType::LParen, "'(' after if");
    s->cond = parseCommaExpression();
    expect(TokenType::RParen, "')'");
    s->thenBranch = parseStatement();
    if (!s->thenBranch) s->thenBranch = std::make_unique<EmptyStmt>();
    if (match(TokenType::KwElse)) {
        s->elseBranch = parseStatement();
        if (!s->elseBranch) s->elseBranch = std::make_unique<EmptyStmt>();
    }
    return s;
}

StmtPtr Parser::parseFor() {
    auto s = std::make_unique<ForStmt>();
    s->loc = get().location;
    expect(TokenType::LParen, "'(' after for");
    localConsts_.emplace_back();
    if (check(TokenType::Semi)) {
        get();
    } else if (isDeclStart()) {
        s->init = parseDeclStmt(true);
    } else {
        auto init = std::make_unique<ExprStmt>();
        init->loc = peek().location;
        init->expr = parseCommaExpression();
        s->init = std::move(init);
        expect(TokenType::Semi, "';'");
    }
    if (!check(TokenType::Semi)) s->cond = parseCommaExpression();
    expect(TokenType::Semi, "';'");
    if (!check(TokenType::RParen)) s->inc = parseCommaExpression();
    expect(TokenType::RParen, "')'");
    s->body = parseStatement();
    if (!s->body) s->body = std::make_unique<EmptyStmt>();
    localConsts_.pop_back();
    return s;
}

StmtPtr Parser::parseWhile() {
    auto s = std::make_unique<WhileStmt>();
    s->loc = get().location;
    expect(TokenType::LParen, "'(' after while");
    s->cond = parseCommaExpression();
    expect(TokenType::RParen, "')'");
    s->body = parseStatement();
    if (!s->body) s->body = std::make_unique<EmptyStmt>();
    return s;
}

StmtPtr Parser::parseDoWhile() {
    auto s = std::make_unique<DoWhileStmt>();
    s->loc = get().location;
    s->body = parseStatement();
    if (!s->body) s->body = std::make_unique<EmptyStmt>();
    if (!expect(TokenType::KwWhile, "'while' after the do body")) return s;
    expect(TokenType::LParen, "'(' after while");
    s->cond = parseCommaExpression();
    expect(TokenType::RParen, "')'");
    expect(TokenType::Semi, "';'");
    return s;
}

StmtPtr Parser::parseSwitch() {
    auto s = std::make_unique<SwitchStmt>();
    s->loc = get().location;
    expect(TokenType::LParen, "'(' after switch");
    s->cond = parseCommaExpression();
    expect(TokenType::RParen, "')'");
    if (!expect(TokenType::LBrace, "'{' after switch(...)")) return s;
    localConsts_.emplace_back();
    while (!check(TokenType::RBrace) && !atEnd()) {
        const std::size_t before = pos_;
        if (check(TokenType::KwCase) || check(TokenType::KwDefault)) {
            SwitchCase c;
            c.loc = peek().location;
            if (match(TokenType::KwDefault)) {
                c.isDefault = true;
            } else {
                get(); // case
                // The label extends up to the ':' that belongs neither to a ?: nor to a parenthesis.
                std::size_t end = pos_;
                int depth = 0, ternary = 0;
                for (; end < toks_.size() && toks_[end].type != TokenType::EndOfFile; ++end) {
                    const TokenType k = toks_[end].type;
                    if (k == TokenType::LParen) ++depth;
                    else if (k == TokenType::RParen) --depth;
                    else if (k == TokenType::Question) ++ternary;
                    else if (k == TokenType::Colon && depth == 0) {
                        if (ternary == 0) break;
                        --ternary;
                    }
                }
                std::int64_t v = 0;
                if (!evalConst(pos_, end, v)) {
                    diags_.reportError(DiagCode::CGFX0019, c.loc, "A case label must be an integer constant");
                }
                c.value = v;
                pos_ = end;
            }
            if (!expect(TokenType::Colon, "':' after the label")) synchronize();
            s->cases.push_back(std::move(c));
        } else {
            StmtPtr st = parseStatement();
            if (panic_) synchronize();
            if (st) {
                if (s->cases.empty()) {
                    errorAt(st->loc, "Statement before the first 'case' of a switch");
                    panic_ = false;
                } else {
                    s->cases.back().body.push_back(std::move(st));
                }
            }
        }
        if (pos_ == before) get();
    }
    localConsts_.pop_back();
    expect(TokenType::RBrace, "'}' closing the switch");
    return s;
}

StmtPtr Parser::parseReturn() {
    auto s = std::make_unique<ReturnStmt>();
    s->loc = get().location;
    if (!check(TokenType::Semi)) s->value = parseCommaExpression();
    expect(TokenType::Semi, "';'");
    return s;
}

// ---------------------------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------------------------

ExprPtr Parser::makeErrorExpr(const SourceLocation& loc) {
    auto e = std::make_unique<LiteralExpr>();
    e->loc = loc;
    e->lit = LiteralExpr::Lit::Int;
    e->text = "0";
    e->type = Type::makeUnknown();
    return e;
}

ExprPtr Parser::parseCommaExpression() {
    ExprPtr lhs = parseExpression(1);
    while (check(TokenType::Comma)) {
        const Token comma = get();
        auto bin = std::make_unique<BinaryExpr>();
        bin->loc = comma.location;
        bin->op = BinaryOp::Comma;
        bin->left = std::move(lhs);
        bin->right = parseExpression(1);
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parseExpression(int minPrec) {
    ExprPtr lhs = parseUnary();
    while (true) {
        const Token& t = peek();
        if (t.type == TokenType::Question) {
            if (2 < minPrec) break;
            auto tern = std::make_unique<TernaryExpr>();
            tern->loc = t.location;
            get();
            tern->cond = std::move(lhs);
            tern->thenExpr = parseExpression(1);
            expect(TokenType::Colon, "':' in the ternary operator");
            tern->elseExpr = parseExpression(2);
            lhs = std::move(tern);
            continue;
        }
        BinaryOp op;
        int prec = 0;
        bool rightAssoc = false;
        if (!binaryOpFor(t.type, op, prec, rightAssoc) || prec < minPrec) break;
        const SourceLocation opLoc = t.location;
        get();
        ExprPtr rhs = parseExpression(rightAssoc ? prec : prec + 1);
        auto bin = std::make_unique<BinaryExpr>();
        bin->loc = opLoc;
        bin->op = op;
        bin->left = std::move(lhs);
        bin->right = std::move(rhs);
        lhs = std::move(bin);
    }
    return lhs;
}

ExprPtr Parser::parseUnary() {
    const Token& t = peek();
    auto makeUnary = [&](UnaryOp op) {
        auto u = std::make_unique<UnaryExpr>();
        u->loc = get().location;
        u->op = op;
        u->operand = parseUnary();
        return u;
    };
    switch (t.type) {
        case TokenType::Plus: return makeUnary(UnaryOp::Plus);
        case TokenType::Minus: return makeUnary(UnaryOp::Minus);
        case TokenType::Bang: return makeUnary(UnaryOp::LogNot);
        case TokenType::Tilde: return makeUnary(UnaryOp::BitNot);
        case TokenType::PlusPlus: return makeUnary(UnaryOp::PreInc);
        case TokenType::MinusMinus: return makeUnary(UnaryOp::PreDec);
        case TokenType::Star: return makeUnary(UnaryOp::Deref);
        case TokenType::Amp: return makeUnary(UnaryOp::AddressOf);
        default: break;
    }
    if (isCastStart()) {
        auto cast = std::make_unique<CastExpr>();
        cast->loc = get().location; // '('
        cast->type = parseType();
        expect(TokenType::RParen, "')' after the cast type");
        cast->operand = parseUnary();
        return cast;
    }
    return parsePostfix(parsePrimary());
}

ExprPtr Parser::parsePostfix(ExprPtr base) {
    while (true) {
        if (check(TokenType::LBracket)) {
            auto idx = std::make_unique<IndexExpr>();
            idx->loc = get().location;
            idx->base = std::move(base);
            idx->index = parseCommaExpression();
            expect(TokenType::RBracket, "']'");
            base = std::move(idx);
        } else if (check(TokenType::Dot) || check(TokenType::Arrow)) {
            const bool arrow = check(TokenType::Arrow);
            const SourceLocation loc = get().location;
            if (!check(TokenType::Identifier)) {
                errorAt(peek().location, "Member name expected after '.'");
                return base;
            }
            const std::string member = get().text;
            if (!arrow && base->kind == ExprKind::Builtin) {
                auto* b = static_cast<BuiltinExpr*>(base.get());
                if (b->component == 0 && (member == "x" || member == "y" || member == "z")) {
                    b->component = member[0];
                    continue;
                }
                diags_.reportError(DiagCode::CGFX0019, loc, "Only the x, y and z components exist for this built-in (found '." + member + "')");
                continue;
            }
            auto m = std::make_unique<MemberExpr>();
            m->loc = loc;
            m->base = std::move(base);
            m->member = member;
            m->viaArrow = arrow;
            base = std::move(m);
        } else if (check(TokenType::PlusPlus) || check(TokenType::MinusMinus)) {
            auto u = std::make_unique<UnaryExpr>();
            u->op = check(TokenType::PlusPlus) ? UnaryOp::PostInc : UnaryOp::PostDec;
            u->loc = get().location;
            u->operand = std::move(base);
            base = std::move(u);
        } else {
            break;
        }
    }
    return base;
}

ExprPtr Parser::parsePrimary() {
    const Token t = peek();
    switch (t.type) {
        case TokenType::IntLiteral:
        case TokenType::FloatLiteral: {
            get();
            auto lit = std::make_unique<LiteralExpr>();
            lit->loc = t.location;
            lit->text = t.text;
            const NumericLiteral n = parseNumericLiteral(t.text);
            switch (n.kind) {
                case NumericLiteral::Kind::Int: lit->lit = LiteralExpr::Lit::Int; lit->intValue = static_cast<std::int64_t>(n.intValue); break;
                case NumericLiteral::Kind::UInt: lit->lit = LiteralExpr::Lit::UInt; lit->intValue = static_cast<std::int64_t>(n.intValue); break;
                case NumericLiteral::Kind::Float: lit->lit = LiteralExpr::Lit::Float; lit->floatValue = n.floatValue; break;
                default: lit->lit = LiteralExpr::Lit::Unsupported; break;
            }
            return lit;
        }
        case TokenType::CharLiteral: {
            get();
            auto lit = std::make_unique<LiteralExpr>();
            lit->loc = t.location;
            lit->text = t.text;
            std::int64_t v = 0;
            if (parseCharLiteral(t.text, v)) {
                lit->lit = LiteralExpr::Lit::Int;
                lit->intValue = v;
            } else {
                lit->lit = LiteralExpr::Lit::Unsupported;
            }
            return lit;
        }
        case TokenType::StringLiteral: {
            get();
            auto lit = std::make_unique<LiteralExpr>();
            lit->loc = t.location;
            lit->text = t.text;
            lit->lit = LiteralExpr::Lit::Unsupported;
            return lit;
        }
        case TokenType::KwTrue:
        case TokenType::KwFalse: {
            get();
            auto lit = std::make_unique<LiteralExpr>();
            lit->loc = t.location;
            lit->text = t.text;
            lit->lit = LiteralExpr::Lit::Bool;
            lit->intValue = t.type == TokenType::KwTrue ? 1 : 0;
            return lit;
        }
        case TokenType::KwSizeof: {
            get();
            auto lit = std::make_unique<LiteralExpr>();
            lit->loc = t.location;
            lit->lit = LiteralExpr::Lit::UInt;
            expect(TokenType::LParen, "'(' after sizeof");
            const Type ty = parseType();
            expect(TokenType::RParen, "')'");
            unsigned size = 0;
            if (ty.isPointer()) size = 8;
            else if (ty.kind == Type::Kind::Bool) size = 1;
            else size = ty.sizeInBytes();
            if (ty.isVector()) size = ty.componentCount() * 4;
            lit->intValue = size;
            lit->text = std::to_string(size) + "u";
            return lit;
        }
        case TokenType::LParen: {
            get();
            auto p = std::make_unique<ParenExpr>();
            p->loc = t.location;
            p->sub = parseCommaExpression();
            expect(TokenType::RParen, "')'");
            return p;
        }
        case TokenType::KwFloat:
        case TokenType::KwInt:
        case TokenType::KwUint:
        case TokenType::KwUnsigned:
        case TokenType::KwBool:
        case TokenType::KwVecType: {
            // float(x), int(x), unsigned(x)...: functional cast; float4(...): call (rejected by Sema)
            if (peek(1).type == TokenType::LParen) {
                if (t.type == TokenType::KwVecType) {
                    get();
                    auto call = std::make_unique<CallExpr>();
                    call->loc = t.location;
                    call->callee = t.text;
                    get(); // (
                    if (!check(TokenType::RParen)) {
                        do { call->args.push_back(parseExpression(1)); } while (match(TokenType::Comma));
                    }
                    expect(TokenType::RParen, "')'");
                    return call;
                }
                auto cast = std::make_unique<CastExpr>();
                cast->loc = t.location;
                cast->type = parseType(false);
                expect(TokenType::LParen, "'('");
                cast->operand = parseExpression(1);
                expect(TokenType::RParen, "')'");
                return cast;
            }
            break;
        }
        case TokenType::Identifier: {
            get();
            std::string name = t.text;
            while (check(TokenType::ColonColon) && peek(1).type == TokenType::Identifier) {
                get();
                name += "::" + get().text;
            }
            if (check(TokenType::TripleLess)) {
                unsupported(t.location, "Kernel launch <<<...>>> from device code: dynamic parallelism is not supported");
                while (!atEnd() && !check(TokenType::TripleGreater) && !check(TokenType::Semi) && !check(TokenType::RBrace)) get();
                if (match(TokenType::TripleGreater) && check(TokenType::LParen)) skipBalanced(TokenType::LParen, TokenType::RParen);
                return makeErrorExpr(t.location);
            }
            BuiltinExpr::Kind bk;
            if (name == t.text && isBuiltinName(name, bk)) {
                auto b = std::make_unique<BuiltinExpr>();
                b->loc = t.location;
                b->builtin = bk;
                return b;
            }
            if (check(TokenType::LParen)) {
                auto call = std::make_unique<CallExpr>();
                call->loc = t.location;
                call->callee = name;
                get();
                if (!check(TokenType::RParen)) {
                    do { call->args.push_back(parseExpression(1)); } while (match(TokenType::Comma));
                }
                expect(TokenType::RParen, "')' after the arguments");
                return call;
            }
            auto ref = std::make_unique<VarRefExpr>();
            ref->loc = t.location;
            ref->name = name;
            return ref;
        }
        case TokenType::LBracket:
            unsupported(t.location, "Lambdas are not supported in device code");
            skipBalanced(TokenType::LBracket, TokenType::RBracket);
            return makeErrorExpr(t.location);
        case TokenType::LBrace:
            unsupported(t.location, "Initializer list {...} is not supported in an expression");
            skipBalanced(TokenType::LBrace, TokenType::RBrace);
            return makeErrorExpr(t.location);
        default:
            break;
    }
    if (isUnsupportedKeyword(t.type)) {
        unsupported(t.location, "C++ construct not supported in device code: '" + t.text + "'");
        get();
        return makeErrorExpr(t.location);
    }
    errorAt(t.location, "Expression expected, found '" + t.text + "'");
    if (!atEnd() && !check(TokenType::Semi) && !check(TokenType::RBrace) && !check(TokenType::RParen)) get();
    return makeErrorExpr(t.location);
}

} // namespace cudagfx

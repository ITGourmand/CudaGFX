// Lexer.cpp
#include "Lexer.hpp"

#include <cctype>
#include <unordered_map>
#include "Literal.hpp"

namespace cudagfx {

namespace {

const std::unordered_map<std::string, TokenType>& keywordTable() {
    static const std::unordered_map<std::string, TokenType> table = {
        {"__global__", TokenType::KwGlobal}, {"__device__", TokenType::KwDevice},
        {"__host__", TokenType::KwHost}, {"__shared__", TokenType::KwShared},
        {"__constant__", TokenType::KwConstant}, {"__launch_bounds__", TokenType::KwLaunchBounds},
        {"__syncthreads", TokenType::KwSyncthreads},

        {"void", TokenType::KwVoid}, {"bool", TokenType::KwBool}, {"char", TokenType::KwChar},
        {"short", TokenType::KwShort}, {"int", TokenType::KwInt}, {"unsigned", TokenType::KwUnsigned},
        {"signed", TokenType::KwSigned}, {"long", TokenType::KwLong}, {"float", TokenType::KwFloat},
        {"double", TokenType::KwDouble}, {"uint", TokenType::KwUint},
        {"float2", TokenType::KwVecType}, {"float3", TokenType::KwVecType}, {"float4", TokenType::KwVecType},
        {"int2", TokenType::KwVecType}, {"int3", TokenType::KwVecType}, {"int4", TokenType::KwVecType},
        {"uint2", TokenType::KwVecType}, {"uint3", TokenType::KwVecType}, {"uint4", TokenType::KwVecType},

        {"if", TokenType::KwIf}, {"else", TokenType::KwElse}, {"for", TokenType::KwFor},
        {"while", TokenType::KwWhile}, {"do", TokenType::KwDo}, {"switch", TokenType::KwSwitch},
        {"case", TokenType::KwCase}, {"default", TokenType::KwDefault}, {"break", TokenType::KwBreak},
        {"continue", TokenType::KwContinue}, {"return", TokenType::KwReturn},

        {"const", TokenType::KwConst}, {"constexpr", TokenType::KwConstexpr}, {"static", TokenType::KwStatic},
        {"inline", TokenType::KwInline}, {"__forceinline__", TokenType::KwInline},
        {"__noinline__", TokenType::KwInline}, {"volatile", TokenType::KwVolatile},
        {"__restrict__", TokenType::KwRestrict}, {"__restrict", TokenType::KwRestrict},
        {"extern", TokenType::KwExtern}, {"struct", TokenType::KwStruct}, {"sizeof", TokenType::KwSizeof},
        {"true", TokenType::KwTrue}, {"false", TokenType::KwFalse}, {"namespace", TokenType::KwNamespace},

        {"template", TokenType::KwTemplate}, {"virtual", TokenType::KwVirtual}, {"try", TokenType::KwTry},
        {"catch", TokenType::KwCatch}, {"throw", TokenType::KwThrow}, {"new", TokenType::KwNew},
        {"delete", TokenType::KwDelete}, {"goto", TokenType::KwGoto}, {"class", TokenType::KwClass},
        {"auto", TokenType::KwAuto}, {"using", TokenType::KwUsing}, {"union", TokenType::KwUnion},
        {"typedef", TokenType::KwTypedef}, {"enum", TokenType::KwEnum}, {"operator", TokenType::KwOperator},
    };
    return table;
}

bool isIdentStart(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return std::isalpha(u) || c == '_' || u >= 0x80;
}

bool isIdentChar(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return std::isalnum(u) || c == '_' || u >= 0x80;
}

TokenType directiveType(const std::string& name) {
    if (name == "include") return TokenType::PpInclude;
    if (name == "define") return TokenType::PpDefine;
    if (name == "undef") return TokenType::PpUndef;
    if (name == "if") return TokenType::PpIf;
    if (name == "ifdef") return TokenType::PpIfdef;
    if (name == "ifndef") return TokenType::PpIfndef;
    if (name == "elif") return TokenType::PpElif;
    if (name == "else") return TokenType::PpElse;
    if (name == "endif") return TokenType::PpEndif;
    if (name == "pragma") return TokenType::PpPragma;
    if (name == "line") return TokenType::PpLine;
    return TokenType::PpOther;
}

std::string describeChar(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    if (u >= 0x20 && u < 0x7F) return std::string("'") + c + "'";
    static const char* hex = "0123456789ABCDEF";
    std::string s = "0x";
    s += hex[u >> 4];
    s += hex[u & 0xF];
    return s;
}

} // namespace

Lexer::Lexer(FileId file, const std::string& source, DiagnosticsEngine& diags)
    : file_(file), source_(source), diags_(diags) {}

std::size_t Lexer::skipSplicesAt(std::size_t pos) const {
    const std::size_t n = source_.size();
    while (pos < n && source_[pos] == '\\') {
        std::size_t p = pos + 1;
        if (p < n && source_[p] == '\r') ++p;
        if (p < n && source_[p] == '\n') pos = p + 1;
        else break;
    }
    return pos;
}

void Lexer::consumeSplices() {
    while (true) {
        const std::size_t next = skipSplicesAt(cursor_);
        if (next == cursor_) return;
        cursor_ = next;
        ++line_;
        column_ = 1;
    }
}

bool Lexer::atEnd() const { return skipSplicesAt(cursor_) >= source_.size(); }

char Lexer::peek(std::size_t ahead) const {
    std::size_t pos = skipSplicesAt(cursor_);
    for (std::size_t i = 0; i < ahead && pos < source_.size(); ++i) pos = skipSplicesAt(pos + 1);
    return pos < source_.size() ? source_[pos] : '\0';
}

char Lexer::get() {
    consumeSplices();
    if (cursor_ >= source_.size()) return '\0';
    const char c = source_[cursor_++];
    if (c == '\n') {
        ++line_;
        column_ = 1;
    } else {
        ++column_;
    }
    return c;
}

SourceLocation Lexer::here() const {
    // The location of a token is that of its first real character (after any line continuations).
    SourceLocation loc;
    loc.file = file_;
    std::size_t pos = cursor_;
    unsigned line = line_;
    unsigned col = column_;
    while (true) {
        const std::size_t next = skipSplicesAt(pos);
        if (next == pos) break;
        pos = next;
        ++line;
        col = 1;
    }
    loc.line = line;
    loc.column = col;
    loc.offset = pos;
    return loc;
}

Token Lexer::makeToken(TokenType type, std::string text, const SourceLocation& start) const {
    Token tok;
    tok.type = type;
    tok.text = std::move(text);
    tok.location = start;
    tok.length = static_cast<unsigned>(cursor_ - start.offset);
    return tok;
}

void Lexer::skipLineComment() {
    // Line continuations are invisible to get(): a "// ... \" comment really does continue on the next line.
    while (!atEnd() && peek() != '\n') get();
}

void Lexer::skipBlockComment() {
    const SourceLocation start = here();
    get(); // '/'
    get(); // '*'
    while (!atEnd()) {
        if (peek() == '*' && peek(1) == '/') {
            get();
            get();
            return;
        }
        get();
    }
    diags_.reportError(DiagCode::CGFX0018, start, "Unterminated block comment");
}

void Lexer::skipTrivia(bool& sawSpace) {
    while (!atEnd()) {
        const char c = peek();
        if (c == '\n') {
            get();
            lineStart_ = true;
            sawSpace = true;
        } else if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v') {
            get();
            sawSpace = true;
        } else if (c == '/' && peek(1) == '/') {
            skipLineComment();
            sawSpace = true;
        } else if (c == '/' && peek(1) == '*') {
            // A multi-line block comment does not reset the "start of line" flag: it counts as a single space.
            const bool wasLineStart = lineStart_;
            skipBlockComment();
            lineStart_ = wasLineStart;
            sawSpace = true;
        } else {
            return;
        }
    }
}

Token Lexer::lexIdentifierOrKeyword(const SourceLocation& start) {
    std::string text;
    while (!atEnd() && isIdentChar(peek())) text += get();

    // Raw strings: R"d( ... )d" (with optional L, u, U, u8 prefixes).
    if (peek() == '"' && !text.empty() && text.back() == 'R' &&
        (text == "R" || text == "LR" || text == "uR" || text == "UR" || text == "u8R")) {
        return lexRawString(start, text);
    }

    const auto& table = keywordTable();
    const auto it = table.find(text);
    if (it != table.end()) return makeToken(it->second, std::move(text), start);
    return makeToken(TokenType::Identifier, std::move(text), start);
}

Token Lexer::lexNumber(const SourceLocation& start) {
    std::string text;
    const bool isHex = peek() == '0' && (peek(1) == 'x' || peek(1) == 'X');
    while (!atEnd()) {
        const char c = peek();
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.') {
            text += get();
            const bool exponent = isHex ? (c == 'p' || c == 'P') : (c == 'e' || c == 'E');
            if (exponent && (peek() == '+' || peek() == '-')) text += get();
        } else if (c == '\'' && std::isalnum(static_cast<unsigned char>(peek(1)))) {
            text += get(); // C++14 digit separator
        } else {
            break;
        }
    }

    const NumericLiteral lit = parseNumericLiteral(text);
    if (lit.kind == NumericLiteral::Kind::Invalid) {
        diags_.reportError(DiagCode::CGFX0018, start, "Invalid numeric literal: '" + text + "'");
        Token tok = makeToken(TokenType::IntLiteral, "0", start);
        return tok;
    }
    return makeToken(lit.kind == NumericLiteral::Kind::Float ? TokenType::FloatLiteral : TokenType::IntLiteral,
                     std::move(text), start);
}

Token Lexer::lexString(const SourceLocation& start) {
    std::string text;
    text += get(); // opening quote
    bool closed = false;
    while (!atEnd()) {
        const char c = peek();
        if (c == '\n') break;
        text += get();
        if (c == '\\') {
            if (!atEnd() && peek() != '\n') text += get();
        } else if (c == '"') {
            closed = true;
            break;
        }
    }
    if (!closed) diags_.reportError(DiagCode::CGFX0018, start, "Unterminated string literal");
    return makeToken(TokenType::StringLiteral, std::move(text), start);
}

Token Lexer::lexRawString(const SourceLocation& start, std::string prefix) {
    std::string text = std::move(prefix);
    text += get(); // quote
    std::string delim;
    bool wellFormed = false;
    while (!atEnd() && delim.size() <= 16) {
        const char c = peek();
        if (c == '(') {
            wellFormed = true;
            break;
        }
        if (c == '\n' || c == ' ' || c == ')' || c == '\\') break;
        delim += c;
        text += get();
    }
    if (!wellFormed) {
        diags_.reportError(DiagCode::CGFX0018, start, "Malformed raw string");
        return makeToken(TokenType::StringLiteral, std::move(text), start);
    }
    text += get(); // '('
    const std::string closing = ")" + delim + "\"";
    bool closed = false;
    while (!atEnd()) {
        text += get();
        if (text.size() >= closing.size() && text.compare(text.size() - closing.size(), closing.size(), closing) == 0) {
            closed = true;
            break;
        }
    }
    if (!closed) diags_.reportError(DiagCode::CGFX0018, start, "Unterminated raw string");
    return makeToken(TokenType::StringLiteral, std::move(text), start);
}

Token Lexer::lexCharLiteral(const SourceLocation& start) {
    std::string text;
    text += get(); // opening apostrophe
    bool closed = false;
    while (!atEnd()) {
        const char c = peek();
        if (c == '\n') break;
        text += get();
        if (c == '\\') {
            if (!atEnd() && peek() != '\n') text += get();
        } else if (c == '\'') {
            closed = true;
            break;
        }
    }
    if (!closed) diags_.reportError(DiagCode::CGFX0018, start, "Unterminated character literal");
    return makeToken(TokenType::CharLiteral, std::move(text), start);
}

Token Lexer::lexDirective(const SourceLocation& start) {
    get(); // '#'
    while (!atEnd() && (peek() == ' ' || peek() == '\t')) get();
    std::string name;
    while (!atEnd() && std::isalpha(static_cast<unsigned char>(peek()))) name += get();
    Token tok = makeToken(directiveType(name), "#" + name, start);
    if (tok.type == TokenType::PpInclude) expectHeader_ = true;
    return tok;
}

Token Lexer::lexHeaderName(const SourceLocation& start) {
    std::string text;
    text += get(); // '<'
    bool closed = false;
    while (!atEnd() && peek() != '\n') {
        const char c = get();
        text += c;
        if (c == '>') {
            closed = true;
            break;
        }
    }
    if (!closed) diags_.reportError(DiagCode::CGFX0014, start, "Unterminated #include header name (expected '>')");
    return makeToken(TokenType::HeaderName, std::move(text), start);
}

Token Lexer::lexOperator(const SourceLocation& start) {
    const char c0 = peek();
    const char c1 = peek(1);
    const char c2 = peek(2);
    auto take = [&](std::size_t n, TokenType type) {
        std::string text;
        for (std::size_t i = 0; i < n; ++i) text += get();
        return makeToken(type, std::move(text), start);
    };

    switch (c0) {
        case '(': return take(1, TokenType::LParen);
        case ')': return take(1, TokenType::RParen);
        case '{': return take(1, TokenType::LBrace);
        case '}': return take(1, TokenType::RBrace);
        case '[': return take(1, TokenType::LBracket);
        case ']': return take(1, TokenType::RBracket);
        case ';': return take(1, TokenType::Semi);
        case ',': return take(1, TokenType::Comma);
        case '?': return take(1, TokenType::Question);
        case '~': return take(1, TokenType::Tilde);
        case ':': return c1 == ':' ? take(2, TokenType::ColonColon) : take(1, TokenType::Colon);
        case '.':
            if (c1 == '.' && c2 == '.') return take(3, TokenType::Ellipsis);
            return take(1, TokenType::Dot);
        case '+':
            if (c1 == '+') return take(2, TokenType::PlusPlus);
            if (c1 == '=') return take(2, TokenType::PlusAssign);
            return take(1, TokenType::Plus);
        case '-':
            if (c1 == '-') return take(2, TokenType::MinusMinus);
            if (c1 == '=') return take(2, TokenType::MinusAssign);
            if (c1 == '>') return take(2, TokenType::Arrow);
            return take(1, TokenType::Minus);
        case '*': return c1 == '=' ? take(2, TokenType::StarAssign) : take(1, TokenType::Star);
        case '/': return c1 == '=' ? take(2, TokenType::SlashAssign) : take(1, TokenType::Slash);
        case '%': return c1 == '=' ? take(2, TokenType::PercentAssign) : take(1, TokenType::Percent);
        case '^': return c1 == '=' ? take(2, TokenType::CaretAssign) : take(1, TokenType::Caret);
        case '!': return c1 == '=' ? take(2, TokenType::NotEqual) : take(1, TokenType::Bang);
        case '=': return c1 == '=' ? take(2, TokenType::Equal) : take(1, TokenType::Assign);
        case '&':
            if (c1 == '&') return take(2, TokenType::AmpAmp);
            if (c1 == '=') return take(2, TokenType::AmpAssign);
            return take(1, TokenType::Amp);
        case '|':
            if (c1 == '|') return take(2, TokenType::PipePipe);
            if (c1 == '=') return take(2, TokenType::PipeAssign);
            return take(1, TokenType::Pipe);
        case '<':
            if (c1 == '<' && c2 == '<') {
                ++openLaunch_;
                return take(3, TokenType::TripleLess);
            }
            if (c1 == '<') return peek(2) == '=' ? take(3, TokenType::LessLessAssign) : take(2, TokenType::LessLess);
            if (c1 == '=') return take(2, TokenType::LessEqual);
            return take(1, TokenType::Less);
        case '>':
            // ">>>" closes a launch only if one is open; otherwise it is ">>" followed by ">"
            // (nested templates in host code: std::vector<std::vector<std::vector<int>>>).
            if (c1 == '>' && c2 == '>' && openLaunch_ > 0) {
                --openLaunch_;
                return take(3, TokenType::TripleGreater);
            }
            if (c1 == '>') return c2 == '=' ? take(3, TokenType::GreaterGreaterAssign) : take(2, TokenType::GreaterGreater);
            if (c1 == '=') return take(2, TokenType::GreaterEqual);
            return take(1, TokenType::Greater);
        case '#':
            return c1 == '#' ? take(2, TokenType::HashHash) : take(1, TokenType::Hash);
        default:
            break;
    }

    // Unexpected character: report a single diagnostic, then advance (no hang is possible).
    const std::string text(1, get());
    diags_.reportError(DiagCode::CGFX0018, start, "Unexpected character " + describeChar(text[0]));
    return makeToken(TokenType::Unknown, text, start);
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    tokens.reserve(source_.size() / 4 + 16);

    // A possible UTF-8 BOM is ignored.
    if (source_.size() >= 3 && static_cast<unsigned char>(source_[0]) == 0xEF &&
        static_cast<unsigned char>(source_[1]) == 0xBB && static_cast<unsigned char>(source_[2]) == 0xBF) {
        cursor_ = 3;
    }

    while (true) {
        bool sawSpace = false;
        skipTrivia(sawSpace);
        consumeSplices();
        if (atEnd()) break;

        const SourceLocation start = here();
        const bool wasHeaderExpected = expectHeader_;
        expectHeader_ = false;
        const char c = peek();

        Token tok;
        if (wasHeaderExpected && c == '<') {
            tok = lexHeaderName(start);
        } else if (c == '#' && lineStart_) {
            tok = lexDirective(start);
        } else if (std::isdigit(static_cast<unsigned char>(c)) ||
                   (c == '.' && std::isdigit(static_cast<unsigned char>(peek(1))))) {
            tok = lexNumber(start);
        } else if (isIdentStart(c)) {
            tok = lexIdentifierOrKeyword(start);
        } else if (c == '"') {
            tok = lexString(start);
        } else if (c == '\'') {
            tok = lexCharLiteral(start);
        } else {
            tok = lexOperator(start);
        }

        tok.atLineStart = lineStart_;
        tok.spaceBefore = sawSpace || tokens.empty();
        lineStart_ = false;
        tokens.push_back(std::move(tok));
    }

    Token eof;
    eof.type = TokenType::EndOfFile;
    eof.location = here();
    eof.atLineStart = true;
    tokens.push_back(std::move(eof));
    return tokens;
}

} // namespace cudagfx

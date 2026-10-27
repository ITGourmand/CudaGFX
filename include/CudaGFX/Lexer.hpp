// Lexer.hpp: lexical analysis of C/C++/CUDA (the subset CudaGFX needs)
#pragma once

#include <string>
#include <vector>
#include "Diagnostics.hpp"
#include "Token.hpp"

namespace cudagfx {

class Lexer {
public:
    // `source` must stay valid for the whole lifetime of the Lexer.
    Lexer(FileId file, const std::string& source, DiagnosticsEngine& diags);

    // Produces every token, terminated by EndOfFile. Throws no exception; lexical errors (unterminated
    // comment, string or literal, unexpected character) are reported as CGFX0018.
    std::vector<Token> tokenize();

private:
    // "Logical" cursor: line continuations (backslash + newline) are invisible.
    char peek(std::size_t ahead = 0) const;
    char get();
    bool atEnd() const;
    void consumeSplices();
    std::size_t skipSplicesAt(std::size_t pos) const;

    SourceLocation here() const;
    void skipTrivia(bool& sawSpace);
    void skipLineComment();
    void skipBlockComment();

    Token makeToken(TokenType type, std::string text, const SourceLocation& start) const;
    Token lexIdentifierOrKeyword(const SourceLocation& start);
    Token lexNumber(const SourceLocation& start);
    Token lexString(const SourceLocation& start);
    Token lexRawString(const SourceLocation& start, std::string prefix);
    Token lexCharLiteral(const SourceLocation& start);
    Token lexDirective(const SourceLocation& start);
    Token lexHeaderName(const SourceLocation& start);
    Token lexOperator(const SourceLocation& start);

    FileId file_;
    const std::string& source_;
    DiagnosticsEngine& diags_;
    std::size_t cursor_ = 0;
    unsigned line_ = 1;
    unsigned column_ = 1;
    bool lineStart_ = true;     // no token emitted yet on the current logical line
    bool expectHeader_ = false; // the next token may be a <header-name>
    unsigned openLaunch_ = 0;   // number of <<< not yet closed (>>> is TripleGreater only then)
};

} // namespace cudagfx

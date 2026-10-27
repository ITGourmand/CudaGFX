// Token.cpp
#include "Token.hpp"

namespace cudagfx {

const char* tokenTypeName(TokenType type) {
    switch (type) {
        case TokenType::EndOfFile: return "end of file";
        case TokenType::Unknown: return "unknown character";
        case TokenType::Identifier: return "identifier";
        case TokenType::IntLiteral: return "integer literal";
        case TokenType::FloatLiteral: return "floating-point literal";
        case TokenType::StringLiteral: return "string";
        case TokenType::CharLiteral: return "character";
        case TokenType::HeaderName: return "header name";
        case TokenType::Semi: return "';'";
        case TokenType::Comma: return "','";
        case TokenType::LParen: return "'('";
        case TokenType::RParen: return "')'";
        case TokenType::LBrace: return "'{'";
        case TokenType::RBrace: return "'}'";
        case TokenType::LBracket: return "'['";
        case TokenType::RBracket: return "']'";
        case TokenType::Colon: return "':'";
        case TokenType::TripleLess: return "'<<<'";
        case TokenType::TripleGreater: return "'>>>'";
        default: return isDirective(type) ? "preprocessor directive" : "token";
    }
}

} // namespace cudagfx

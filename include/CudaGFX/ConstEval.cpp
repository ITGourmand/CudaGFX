// ConstEval.cpp
#include "ConstEval.hpp"

#include "Literal.hpp"

namespace cudagfx {

namespace {

struct BinOp {
    TokenType type;
    int prec;
};

// C precedences (larger = binds tighter).
int binaryPrecedence(TokenType t) {
    switch (t) {
        case TokenType::PipePipe: return 1;
        case TokenType::AmpAmp: return 2;
        case TokenType::Pipe: return 3;
        case TokenType::Caret: return 4;
        case TokenType::Amp: return 5;
        case TokenType::Equal: case TokenType::NotEqual: return 6;
        case TokenType::Less: case TokenType::LessEqual:
        case TokenType::Greater: case TokenType::GreaterEqual: return 7;
        case TokenType::LessLess: case TokenType::GreaterGreater: return 8;
        case TokenType::Plus: case TokenType::Minus: return 9;
        case TokenType::Star: case TokenType::Slash: case TokenType::Percent: return 10;
        default: return 0;
    }
}

std::int64_t wrap(std::uint64_t v) { return static_cast<std::int64_t>(v); }

bool isTypeKeyword(TokenType t) {
    switch (t) {
        case TokenType::KwBool: case TokenType::KwChar: case TokenType::KwShort: case TokenType::KwInt:
        case TokenType::KwUnsigned: case TokenType::KwSigned: case TokenType::KwLong:
        case TokenType::KwUint: case TokenType::KwConst:
            return true;
        default:
            return false;
    }
}

} // namespace

bool ConstEvaluator::fail(std::string message) {
    if (error_.empty()) error_ = std::move(message);
    return false;
}

bool ConstEvaluator::evaluate(const Token* first, const Token* last, std::int64_t& out) {
    error_.clear();
    cur_ = first;
    end_ = last;
    if (atEnd()) return fail("empty expression");
    if (!parseTernary(out)) return false;
    if (!atEnd()) return fail("unexpected token '" + tok().text + "' in constant expression");
    return true;
}

bool ConstEvaluator::parseTernary(std::int64_t& out) {
    std::int64_t cond = 0;
    if (!parseBinary(1, cond)) return false;
    if (atEnd() || tok().type != TokenType::Question) {
        out = cond;
        return true;
    }
    ++cur_;
    std::int64_t a = 0, b = 0;
    if (!parseTernary(a)) return false;
    if (atEnd() || tok().type != TokenType::Colon) return fail("expected ':' in the ternary operator");
    ++cur_;
    if (!parseTernary(b)) return false;
    out = cond != 0 ? a : b;
    return true;
}

bool ConstEvaluator::parseBinary(int minPrec, std::int64_t& out) {
    std::int64_t lhs = 0;
    if (!parseUnary(lhs)) return false;
    while (!atEnd()) {
        const TokenType op = tok().type;
        const int prec = binaryPrecedence(op);
        if (prec == 0 || prec < minPrec) break;
        ++cur_;
        std::int64_t rhs = 0;
        if (!parseBinary(prec + 1, rhs)) return false;
        const std::uint64_t ul = static_cast<std::uint64_t>(lhs);
        const std::uint64_t ur = static_cast<std::uint64_t>(rhs);
        switch (op) {
            case TokenType::PipePipe: lhs = (lhs != 0 || rhs != 0); break;
            case TokenType::AmpAmp: lhs = (lhs != 0 && rhs != 0); break;
            case TokenType::Pipe: lhs = wrap(ul | ur); break;
            case TokenType::Caret: lhs = wrap(ul ^ ur); break;
            case TokenType::Amp: lhs = wrap(ul & ur); break;
            case TokenType::Equal: lhs = (lhs == rhs); break;
            case TokenType::NotEqual: lhs = (lhs != rhs); break;
            case TokenType::Less: lhs = (lhs < rhs); break;
            case TokenType::LessEqual: lhs = (lhs <= rhs); break;
            case TokenType::Greater: lhs = (lhs > rhs); break;
            case TokenType::GreaterEqual: lhs = (lhs >= rhs); break;
            case TokenType::LessLess:
                if (rhs < 0 || rhs >= 64) return fail("shift out of range in constant expression");
                lhs = wrap(ul << ur);
                break;
            case TokenType::GreaterGreater:
                if (rhs < 0 || rhs >= 64) return fail("shift out of range in constant expression");
                lhs = lhs >> rhs;
                break;
            case TokenType::Plus: lhs = wrap(ul + ur); break;
            case TokenType::Minus: lhs = wrap(ul - ur); break;
            case TokenType::Star: lhs = wrap(ul * ur); break;
            case TokenType::Slash:
                if (rhs == 0) return fail("division by zero in constant expression");
                lhs = (lhs == INT64_MIN && rhs == -1) ? lhs : lhs / rhs;
                break;
            case TokenType::Percent:
                if (rhs == 0) return fail("modulo by zero in constant expression");
                lhs = (rhs == -1) ? 0 : lhs % rhs;
                break;
            default:
                return fail("unexpected operator");
        }
    }
    out = lhs;
    return true;
}

bool ConstEvaluator::parseUnary(std::int64_t& out) {
    if (atEnd()) return fail("incomplete expression");
    switch (tok().type) {
        case TokenType::Plus:
            ++cur_;
            return parseUnary(out);
        case TokenType::Minus: {
            ++cur_;
            std::int64_t v = 0;
            if (!parseUnary(v)) return false;
            out = wrap(0 - static_cast<std::uint64_t>(v));
            return true;
        }
        case TokenType::Bang: {
            ++cur_;
            std::int64_t v = 0;
            if (!parseUnary(v)) return false;
            out = (v == 0);
            return true;
        }
        case TokenType::Tilde: {
            ++cur_;
            std::int64_t v = 0;
            if (!parseUnary(v)) return false;
            out = ~v;
            return true;
        }
        default:
            return parsePrimary(out);
    }
}

bool ConstEvaluator::parsePrimary(std::int64_t& out) {
    const Token& t = tok();
    switch (t.type) {
        case TokenType::IntLiteral: {
            const NumericLiteral lit = parseNumericLiteral(t.text);
            if (lit.kind != NumericLiteral::Kind::Int && lit.kind != NumericLiteral::Kind::UInt &&
                lit.kind != NumericLiteral::Kind::Int64) {
                return fail("invalid integer literal '" + t.text + "'");
            }
            out = wrap(lit.intValue);
            ++cur_;
            return true;
        }
        case TokenType::CharLiteral: {
            if (!parseCharLiteral(t.text, out)) return fail("unsupported character literal " + t.text);
            ++cur_;
            return true;
        }
        case TokenType::KwTrue:
            out = 1;
            ++cur_;
            return true;
        case TokenType::KwFalse:
            out = 0;
            ++cur_;
            return true;
        case TokenType::FloatLiteral:
            return fail("floating-point literal '" + t.text + "' in an expression that must be an integer");
        case TokenType::KwSizeof: {
            ++cur_;
            if (atEnd() || tok().type != TokenType::LParen) return fail("expected '(' after sizeof");
            ++cur_;
            if (atEnd()) return fail("expected a type in sizeof");
            std::int64_t size = 0;
            switch (tok().type) {
                case TokenType::KwFloat: case TokenType::KwInt: case TokenType::KwUint:
                case TokenType::KwUnsigned: size = 4; break;
                case TokenType::KwBool: case TokenType::KwChar: size = 1; break;
                case TokenType::KwShort: size = 2; break;
                case TokenType::KwVecType: {
                    const char digit = tok().text.empty() ? '0' : tok().text.back();
                    size = 4 * (digit - '0');
                    break;
                }
                default: return fail("sizeof of this type cannot be evaluated at compile time");
            }
            ++cur_;
            if (atEnd() || tok().type != TokenType::RParen) return fail("expected ')' after sizeof");
            ++cur_;
            out = size;
            return true;
        }
        case TokenType::LParen: {
            ++cur_;
            // Scalar cast: (int) x, (unsigned int) x, (bool) x ...
            if (!atEnd() && isTypeKeyword(tok().type)) {
                bool isBool = false;
                bool isUnsigned = false;
                while (!atEnd() && isTypeKeyword(tok().type)) {
                    if (tok().type == TokenType::KwBool) isBool = true;
                    if (tok().type == TokenType::KwUnsigned || tok().type == TokenType::KwUint) isUnsigned = true;
                    ++cur_;
                }
                if (atEnd() || tok().type != TokenType::RParen) return fail("expected ')' after the cast type");
                ++cur_;
                std::int64_t v = 0;
                if (!parseUnary(v)) return false;
                if (isBool) out = (v != 0);
                else if (isUnsigned) out = static_cast<std::int64_t>(static_cast<std::uint32_t>(v));
                else out = static_cast<std::int64_t>(static_cast<std::int32_t>(v));
                return true;
            }
            if (!parseTernary(out)) return false;
            if (atEnd() || tok().type != TokenType::RParen) return fail("expected ')'");
            ++cur_;
            return true;
        }
        case TokenType::Identifier: {
            std::int64_t v = 0;
            if (resolver_ && resolver_(t.text, v)) {
                out = v;
                ++cur_;
                return true;
            }
            return fail("'" + t.text + "' is not a known compile-time constant");
        }
        default:
            return fail("unexpected token '" + t.text + "' in constant expression");
    }
}

} // namespace cudagfx

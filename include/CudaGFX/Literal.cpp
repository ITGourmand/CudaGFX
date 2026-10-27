// Literal.cpp
#include "Literal.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace cudagfx {

namespace {

std::string stripSeparators(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
        if (c != '\'') out += c;
    }
    return out;
}

bool isHexDigit(char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; }

} // namespace

NumericLiteral parseNumericLiteral(const std::string& rawText) {
    NumericLiteral lit;
    const std::string text = stripSeparators(rawText);
    if (text.empty()) return lit;

    const bool isHex = text.size() > 1 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X');
    const bool isBin = text.size() > 1 && text[0] == '0' && (text[1] == 'b' || text[1] == 'B');

    // Floating-point detection: decimal point, decimal exponent, hexadecimal binary exponent, or an f suffix on a decimal.
    bool isFloat = false;
    if (isHex) {
        isFloat = text.find('.') != std::string::npos || text.find('p') != std::string::npos ||
                  text.find('P') != std::string::npos;
    } else if (!isBin) {
        isFloat = text.find('.') != std::string::npos || text.find('e') != std::string::npos ||
                  text.find('E') != std::string::npos ||
                  text.back() == 'f' || text.back() == 'F';
    }

    if (isFloat) {
        std::string body = text;
        bool floatSuffix = false;
        if (!body.empty() && (body.back() == 'f' || body.back() == 'F') && !isHex) {
            floatSuffix = true;
            body.pop_back();
        } else if (!body.empty() && (body.back() == 'l' || body.back() == 'L')) {
            return lit; // long double: not supported
        }
        char* end = nullptr;
        const double v = std::strtod(body.c_str(), &end);
        if (end == body.c_str() || *end != '\0' || !std::isfinite(v)) return lit;
        if (std::fabs(v) > static_cast<double>(std::numeric_limits<float>::max())) return lit;
        lit.kind = NumericLiteral::Kind::Float;
        lit.floatValue = v;
        lit.hasFloatSuffix = floatSuffix;
        return lit;
    }

    // Integer: prefix, digits, u/l suffixes.
    std::size_t pos = 0;
    int base = 10;
    if (isHex) { base = 16; pos = 2; }
    else if (isBin) { base = 2; pos = 2; }
    else if (text.size() > 1 && text[0] == '0') { base = 8; pos = 1; }

    std::uint64_t value = 0;
    bool overflow = false;
    std::size_t digits = 0;
    for (; pos < text.size(); ++pos) {
        const char c = text[pos];
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (base == 16 && isHexDigit(c)) d = std::tolower(static_cast<unsigned char>(c)) - 'a' + 10;
        else break;
        if (d >= base) return lit; // digit invalid for the base
        if (value > (std::numeric_limits<std::uint64_t>::max() - static_cast<unsigned>(d)) / static_cast<unsigned>(base)) {
            overflow = true;
        }
        value = value * static_cast<unsigned>(base) + static_cast<unsigned>(d);
        ++digits;
    }
    if (digits == 0 && !(base == 8 && text == "0") && !(base == 8)) return lit;

    bool hasU = false;
    unsigned longCount = 0;
    for (; pos < text.size(); ++pos) {
        const char c = text[pos];
        if (c == 'u' || c == 'U') {
            if (hasU) return lit;
            hasU = true;
        } else if (c == 'l' || c == 'L') {
            ++longCount;
            if (longCount > 2) return lit;
        } else {
            return lit;
        }
    }

    lit.intValue = value;
    if (overflow || longCount > 0) {
        lit.kind = NumericLiteral::Kind::Int64;
    } else if (hasU) {
        lit.kind = value <= 0xFFFFFFFFull ? NumericLiteral::Kind::UInt : NumericLiteral::Kind::Int64;
    } else if (value <= 0x7FFFFFFFull) {
        lit.kind = NumericLiteral::Kind::Int;
    } else if (base != 10 && value <= 0xFFFFFFFFull) {
        lit.kind = NumericLiteral::Kind::UInt; // C++ rule: unsuffixed hexadecimal/octal/binary literal
    } else {
        lit.kind = NumericLiteral::Kind::Int64;
    }
    return lit;
}

bool parseCharLiteral(const std::string& text, std::int64_t& value) {
    // Expected form: '<c>' or '\<seq>' (L/u/U prefixes are not supported).
    if (text.size() < 3 || text.front() != '\'' || text.back() != '\'') return false;
    const std::string body = text.substr(1, text.size() - 2);
    if (body.size() == 1 && body[0] != '\\') {
        value = static_cast<unsigned char>(body[0]);
        return true;
    }
    if (body.size() >= 2 && body[0] == '\\') {
        switch (body[1]) {
            case 'n': value = '\n'; return body.size() == 2;
            case 't': value = '\t'; return body.size() == 2;
            case 'r': value = '\r'; return body.size() == 2;
            case 'a': value = '\a'; return body.size() == 2;
            case 'b': value = '\b'; return body.size() == 2;
            case 'f': value = '\f'; return body.size() == 2;
            case 'v': value = '\v'; return body.size() == 2;
            case '\\': value = '\\'; return body.size() == 2;
            case '\'': value = '\''; return body.size() == 2;
            case '"': value = '"'; return body.size() == 2;
            case 'x': {
                if (body.size() < 3) return false;
                char* end = nullptr;
                const long v = std::strtol(body.c_str() + 2, &end, 16);
                if (*end != '\0') return false;
                value = v;
                return true;
            }
            default:
                if (body[1] >= '0' && body[1] <= '7') {
                    char* end = nullptr;
                    const long v = std::strtol(body.c_str() + 1, &end, 8);
                    if (*end != '\0') return false;
                    value = v;
                    return true;
                }
                return false;
        }
    }
    return false;
}

std::string formatFloatLiteral(double value) {
    const float f = static_cast<float>(value);
    char buf[48];
    // Shortest decimal representation that reads back as exactly the same float.
    for (int precision = 1; precision <= 9; ++precision) {
        std::snprintf(buf, sizeof(buf), "%.*g", precision, static_cast<double>(f));
        if (std::strtof(buf, nullptr) == f) break;
    }
    std::string s = buf;
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0"; // 'n': inf/nan, should not happen
    return s;
}

} // namespace cudagfx

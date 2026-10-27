// ConstEval.hpp: evaluation of constant integer expressions over a token sequence.
//
// Used by the preprocessor (#if), the parser (array sizes, case labels, __launch_bounds__) and the
// launch-site analysis (block sizes). One evaluator, one grammar: the full C operator set with the
// correct precedence.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include "Token.hpp"

namespace cudagfx {

class ConstEvaluator {
public:
    // Resolves an identifier to an integer value; returns false if the identifier is not a constant.
    using Resolver = std::function<bool(const std::string& name, std::int64_t& value)>;

    explicit ConstEvaluator(Resolver resolver = {}) : resolver_(std::move(resolver)) {}

    // Evaluates the tokens [first, last). Returns true and sets `out` if the expression is an integer
    // constant; otherwise returns false and error() describes the cause (ready to display).
    bool evaluate(const Token* first, const Token* last, std::int64_t& out);

    const std::string& error() const { return error_; }

private:
    bool parseTernary(std::int64_t& out);
    bool parseBinary(int minPrec, std::int64_t& out);
    bool parseUnary(std::int64_t& out);
    bool parsePrimary(std::int64_t& out);
    bool fail(std::string message);
    bool atEnd() const { return cur_ == end_; }
    const Token& tok() const { return *cur_; }

    Resolver resolver_;
    const Token* cur_ = nullptr;
    const Token* end_ = nullptr;
    std::string error_;
};

} // namespace cudagfx

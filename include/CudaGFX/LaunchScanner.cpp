// LaunchScanner.cpp: finds the <<<grid, block[, shared[, stream]]>>>(args) launches of the host code.
//
// The ACTIVE stream (macros not expanded) is walked once, following the { } scopes to recover the
// host constants (`const int B = 128;`, `dim3 block(16, 16);`) visible at each launch.
// The text of the arguments is rebuilt from the tokens (never from the raw text: comments or
// newlines it might contain must not cross the rewrite).
#include "Parser.hpp"

namespace cudagfx {

namespace {

bool isIntegerTypeKeyword(TokenType t) {
    switch (t) {
        case TokenType::KwInt: case TokenType::KwUnsigned: case TokenType::KwSigned: case TokenType::KwUint:
        case TokenType::KwLong: case TokenType::KwShort: case TokenType::KwChar: case TokenType::KwBool:
            return true;
        default:
            return false;
    }
}

bool isIntegerTypedef(const std::string& s) {
    return s == "size_t" || s == "ssize_t" || s == "ptrdiff_t" || s == "int8_t" || s == "int16_t" ||
           s == "int32_t" || s == "int64_t" || s == "uint8_t" || s == "uint16_t" || s == "uint32_t" ||
           s == "uint64_t";
}

bool isAssignmentToken(TokenType t) {
    switch (t) {
        case TokenType::Assign: case TokenType::PlusAssign: case TokenType::MinusAssign:
        case TokenType::StarAssign: case TokenType::SlashAssign: case TokenType::PercentAssign:
        case TokenType::LessLessAssign: case TokenType::GreaterGreaterAssign: case TokenType::AmpAssign:
        case TokenType::PipeAssign: case TokenType::CaretAssign: case TokenType::PlusPlus:
        case TokenType::MinusMinus:
            return true;
        default:
            return false;
    }
}

bool isOpen(TokenType t) { return t == TokenType::LParen || t == TokenType::LBracket || t == TokenType::LBrace; }
bool isClose(TokenType t) { return t == TokenType::RParen || t == TokenType::RBracket || t == TokenType::RBrace; }

} // namespace

const Parser::HostEntry* Parser::findHostEntry(const std::string& name) const {
    for (auto it = hostScopes_.rbegin(); it != hostScopes_.rend(); ++it) {
        const auto f = it->find(name);
        if (f != it->end()) return &f->second;
    }
    return nullptr;
}

std::string Parser::sliceText(std::size_t firstTok, std::size_t lastTok) const {
    std::string s;
    for (std::size_t i = firstTok; i <= lastTok && i < raw_.size(); ++i) {
        if (!s.empty() && raw_[i].spaceBefore) s += ' ';
        s += raw_[i].text;
    }
    return s;
}

bool Parser::evalHostRange(const std::vector<Token>& expanded, std::int64_t& out, std::string& why) {
    ConstEvaluator ev([this](const std::string& n, std::int64_t& v) {
        const HostEntry* e = findHostEntry(n);
        if (e && e->isConst && e->known && !e->isDim3) {
            v = e->value;
            return true;
        }
        if (e && !e->isConst) lastUnresolved_ = "variable '" + n + "' is not const";
        else if (e && !e->known) lastUnresolved_ = "'" + n + "' is const but its initializer cannot be evaluated";
        else lastUnresolved_ = "identifier '" + n + "' is unknown or not a constant";
        return false;
    });
    lastUnresolved_.clear();
    if (ev.evaluate(expanded.data(), expanded.data() + expanded.size(), out)) return true;
    why = lastUnresolved_.empty() ? ev.error() : lastUnresolved_;
    return false;
}

bool Parser::parseHostDeclaration(std::size_t& i) {
    const std::size_t n = raw_.size();
    std::size_t j = i;
    bool isConst = false;
    while (j < n && (raw_[j].type == TokenType::KwStatic || raw_[j].type == TokenType::KwConst ||
                     raw_[j].type == TokenType::KwConstexpr || raw_[j].type == TokenType::KwVolatile ||
                     raw_[j].type == TokenType::KwInline || raw_[j].type == TokenType::KwExtern)) {
        if (raw_[j].type == TokenType::KwConst || raw_[j].type == TokenType::KwConstexpr) isConst = true;
        ++j;
    }
    bool isDim3 = false;
    if (j < n && raw_[j].type == TokenType::Identifier && raw_[j].text == "dim3") {
        isDim3 = true;
        ++j;
    } else if (j < n && isIntegerTypeKeyword(raw_[j].type)) {
        while (j < n && (isIntegerTypeKeyword(raw_[j].type) || raw_[j].type == TokenType::KwConst)) {
            if (raw_[j].type == TokenType::KwConst) isConst = true;
            ++j;
        }
    } else if (j < n && raw_[j].type == TokenType::Identifier && isIntegerTypedef(raw_[j].text)) {
        ++j;
    } else {
        return false;
    }

    // Declarators: we only commit if the statement ends with ';'.
    std::vector<std::pair<std::string, HostEntry>> found;
    while (true) {
        if (j >= n || raw_[j].type != TokenType::Identifier) return false; // pointer, call, cast...
        const std::string name = raw_[j].text;
        ++j;
        HostEntry e;
        e.isConst = isConst;

        // Delimits the initializer: up to a ',' or ';' at depth 0.
        auto findEnd = [&](std::size_t from) {
            int depth = 0;
            std::size_t k = from;
            for (; k < n; ++k) {
                const TokenType ty = raw_[k].type;
                if (isOpen(ty)) ++depth;
                else if (isClose(ty)) { if (depth == 0) break; --depth; }
                else if (depth == 0 && (ty == TokenType::Comma || ty == TokenType::Semi)) break;
                else if (ty == TokenType::EndOfFile) break;
            }
            return k;
        };

        auto evalRange = [&](std::size_t a, std::size_t b, std::int64_t& v) {
            std::vector<Token> slice(raw_.begin() + static_cast<std::ptrdiff_t>(a), raw_.begin() + static_cast<std::ptrdiff_t>(b));
            const std::vector<Token> ex = pp_.expand(slice);
            std::string why;
            return !ex.empty() && evalHostRange(ex, v, why);
        };

        // Evaluates a list of dim3 arguments (tokens [a,b), separated by commas).
        auto evalDim3Args = [&](std::size_t a, std::size_t b) {
            std::int64_t d[3] = {1, 1, 1};
            std::size_t argStart = a;
            int idx = 0;
            int depth = 0;
            bool ok = true;
            for (std::size_t k = a; k <= b; ++k) {
                const bool atEndTok = (k == b);
                if (!atEndTok) {
                    if (isOpen(raw_[k].type)) ++depth;
                    else if (isClose(raw_[k].type)) --depth;
                }
                if (atEndTok || (depth == 0 && raw_[k].type == TokenType::Comma)) {
                    if (argStart == k) { ok = ok && (a == b); }
                    else if (idx >= 3 || !evalRange(argStart, k, d[idx])) ok = false;
                    ++idx;
                    argStart = k + 1;
                }
            }
            if (ok) {
                e.known = true;
                for (int q = 0; q < 3; ++q) e.dims[q] = d[q];
            }
        };

        if (isDim3) {
            e.isDim3 = true;
            if (j < n && (raw_[j].type == TokenType::LParen || raw_[j].type == TokenType::LBrace)) {
                const TokenType open = raw_[j].type;
                const TokenType close = open == TokenType::LParen ? TokenType::RParen : TokenType::RBrace;
                int depth = 0;
                std::size_t k = j;
                for (; k < n; ++k) {
                    if (raw_[k].type == open) ++depth;
                    else if (raw_[k].type == close && --depth == 0) break;
                }
                if (k >= n) return false;
                evalDim3Args(j + 1, k);
                j = k + 1;
            } else if (j < n && raw_[j].type == TokenType::Assign) {
                ++j;
                const std::size_t end = findEnd(j);
                if (j < n && raw_[j].type == TokenType::Identifier && raw_[j].text == "dim3" && j + 1 < end &&
                    raw_[j + 1].type == TokenType::LParen && raw_[end - 1].type == TokenType::RParen) {
                    evalDim3Args(j + 2, end - 1);
                } else if (j < n && raw_[j].type == TokenType::LBrace && end > j && raw_[end - 1].type == TokenType::RBrace) {
                    evalDim3Args(j + 1, end - 1);
                } else {
                    std::int64_t v = 0;
                    if (evalRange(j, end, v)) {
                        e.known = true;
                        e.dims[0] = v;
                    }
                }
                j = end;
            } else {
                e.known = true; // `dim3 d;` is (1,1,1)
            }
        } else {
            if (j < n && raw_[j].type == TokenType::Assign) {
                ++j;
                const std::size_t end = findEnd(j);
                std::int64_t v = 0;
                if (evalRange(j, end, v)) {
                    e.known = true;
                    e.value = v;
                }
                j = end;
            } else if (j < n && (raw_[j].type == TokenType::LParen || raw_[j].type == TokenType::LBrace)) {
                const TokenType open = raw_[j].type;
                const TokenType close = open == TokenType::LParen ? TokenType::RParen : TokenType::RBrace;
                int depth = 0;
                std::size_t k = j;
                for (; k < n; ++k) {
                    if (raw_[k].type == open) ++depth;
                    else if (raw_[k].type == close && --depth == 0) break;
                }
                if (k >= n) return false;
                std::int64_t v = 0;
                if (evalRange(j + 1, k, v)) {
                    e.known = true;
                    e.value = v;
                }
                j = k + 1;
            }
        }
        found.emplace_back(name, e);
        if (j < n && raw_[j].type == TokenType::Comma) {
            ++j;
            continue;
        }
        break;
    }
    if (j >= n || raw_[j].type != TokenType::Semi) return false;
    for (auto& f : found) hostScopes_.back()[f.first] = f.second;
    i = j;
    return true;
}

bool Parser::evaluateBlock(std::size_t first, std::size_t last, const SourceLocation& at, std::string& text,
                           unsigned dims[3]) {
    text = sliceText(first, last - 1);
    std::vector<Token> slice(raw_.begin() + static_cast<std::ptrdiff_t>(first), raw_.begin() + static_cast<std::ptrdiff_t>(last));
    const std::vector<Token> ex = pp_.expand(slice);

    auto fail = [&](const std::string& reason) {
        diags_.reportError(DiagCode::CGFX0006, at,
                           "The block size must be a compile-time constant: '" + text + "' (" + reason + ")");
        return false;
    };
    if (ex.empty()) return fail("empty expression");

    std::int64_t d[3] = {1, 1, 1};
    if (ex[0].type == TokenType::Identifier && ex[0].text == "dim3" && ex.size() >= 3 &&
        (ex[1].type == TokenType::LParen || ex[1].type == TokenType::LBrace)) {
        const TokenType close = ex[1].type == TokenType::LParen ? TokenType::RParen : TokenType::RBrace;
        if (ex.back().type != close) return fail("unrecognized dim3 form");
        std::size_t argStart = 2;
        int idx = 0;
        int depth = 0;
        for (std::size_t k = 2; k < ex.size(); ++k) {
            const bool atEndTok = (k == ex.size() - 1);
            if (!atEndTok) {
                if (isOpen(ex[k].type)) ++depth;
                else if (isClose(ex[k].type)) --depth;
            }
            if (atEndTok || (depth == 0 && ex[k].type == TokenType::Comma)) {
                if (argStart == k) return fail("empty dim3 argument");
                if (idx >= 3) return fail("dim3 accepts at most 3 arguments");
                std::vector<Token> arg(ex.begin() + static_cast<std::ptrdiff_t>(argStart), ex.begin() + static_cast<std::ptrdiff_t>(k));
                std::string why;
                if (!evalHostRange(arg, d[idx], why)) return fail(why);
                ++idx;
                argStart = k + 1;
            }
        }
    } else if (ex.size() == 1 && ex[0].type == TokenType::Identifier && findHostEntry(ex[0].text) &&
               findHostEntry(ex[0].text)->isDim3) {
        const HostEntry* e = findHostEntry(ex[0].text);
        if (e->modified) return fail("dim3 variable '" + ex[0].text + "' is modified after its declaration");
        if (!e->known) return fail("dim3 variable '" + ex[0].text + "' is not initialized with constants");
        for (int q = 0; q < 3; ++q) d[q] = e->dims[q];
    } else {
        std::string why;
        if (!evalHostRange(ex, d[0], why)) return fail(why);
    }
    for (int q = 0; q < 3; ++q) {
        if (d[q] < 1) return fail("dimension " + std::string(1, static_cast<char>('x' + q)) + " is " + std::to_string(d[q]));
        if (d[q] > 65535) return fail("dimension " + std::string(1, static_cast<char>('x' + q)) + " is out of range (" + std::to_string(d[q]) + ")");
        dims[q] = static_cast<unsigned>(d[q]);
    }
    return true;
}

bool Parser::handleLaunch(std::size_t& i) {
    const std::size_t n = raw_.size();
    const Token& kernelTok = raw_[i];
    std::size_t startIdx = i;
    while (startIdx >= 2 && raw_[startIdx - 1].type == TokenType::ColonColon && raw_[startIdx - 2].type == TokenType::Identifier) {
        startIdx -= 2;
    }
    if (startIdx >= 1 && raw_[startIdx - 1].type == TokenType::ColonColon) startIdx -= 1;

    auto malformed = [&](const std::string& why, std::size_t resume) {
        diags_.reportError(DiagCode::CGFX0013, kernelTok.location, "Malformed kernel launch: " + why);
        i = resume;
        return false;
    };

    // Configuration : <<< a, b[, c[, d]] >>>
    std::vector<std::pair<std::size_t, std::size_t>> cfg;
    std::size_t k = i + 2;
    std::size_t argStart = k;
    int depth = 0;
    std::size_t closeIdx = 0;
    bool closed = false;
    for (; k < n; ++k) {
        const TokenType ty = raw_[k].type;
        if (isOpen(ty)) ++depth;
        else if (isClose(ty)) { if (depth == 0) break; --depth; }
        else if (depth == 0 && ty == TokenType::Comma) { cfg.emplace_back(argStart, k); argStart = k + 1; }
        else if (depth == 0 && ty == TokenType::TripleGreater) { cfg.emplace_back(argStart, k); closeIdx = k; closed = true; break; }
        else if (ty == TokenType::Semi || ty == TokenType::EndOfFile) break;
    }
    if (!closed) return malformed("expected '>>>' (nested templates inside the launch configuration must be parenthesized)", k);
    if (cfg.size() < 2 || cfg.size() > 4) return malformed("<<<...>>> expects 2 to 4 arguments (grid, block[, shared memory[, stream]])", closeIdx);
    for (const auto& c : cfg) {
        if (c.first >= c.second) return malformed("empty launch configuration argument", closeIdx);
    }

    // Call arguments: (a, b, ...)
    std::size_t p = closeIdx + 1;
    if (p >= n || raw_[p].type != TokenType::LParen) return malformed("expected '(' after '>>>'", closeIdx);
    std::vector<std::pair<std::size_t, std::size_t>> callArgs;
    std::size_t aStart = p + 1;
    depth = 0;
    std::size_t closeParen = 0;
    bool parenClosed = false;
    for (std::size_t q = p; q < n; ++q) {
        const TokenType ty = raw_[q].type;
        if (isOpen(ty)) {
            ++depth;
        } else if (isClose(ty)) {
            --depth;
            if (depth == 0) {
                if (aStart < q) callArgs.emplace_back(aStart, q);
                closeParen = q;
                parenClosed = true;
                break;
            }
        } else if (depth == 1 && ty == TokenType::Comma) {
            if (aStart >= q) return malformed("empty call argument", q);
            callArgs.emplace_back(aStart, q);
            aStart = q + 1;
        } else if (ty == TokenType::EndOfFile) {
            break;
        }
    }
    if (!parenClosed) return malformed("expected ')' at the end of the call", n - 1);

    if (kernelTok.location.file != mainFile_) {
        diags_.reportError(DiagCode::CGFX0013, kernelTok.location,
                           "Kernel launch inside an included file: not supported (launch it from the main file)");
        i = closeParen;
        return false;
    }

    LaunchSite site;
    site.kernelName = kernelTok.text;
    site.kernelLoc = kernelTok.location;
    site.span.start = raw_[startIdx].location;
    site.span.end = raw_[closeParen].location;
    site.span.end.offset = raw_[closeParen].endOffset();
    site.gridExprText = sliceText(cfg[0].first, cfg[0].second - 1);

    bool ok = evaluateBlock(cfg[1].first, cfg[1].second, raw_[cfg[1].first].location, site.blockExprText, site.evaluatedBlock);

    if (cfg.size() >= 3) {
        std::vector<Token> slice(raw_.begin() + static_cast<std::ptrdiff_t>(cfg[2].first), raw_.begin() + static_cast<std::ptrdiff_t>(cfg[2].second));
        const std::vector<Token> ex = pp_.expand(slice);
        std::int64_t v = 0;
        std::string why;
        if (ex.empty() || !evalHostRange(ex, v, why) || v != 0) {
            diags_.reportError(DiagCode::CGFX0004, raw_[cfg[2].first].location,
                               "Dynamic shared memory (3rd argument of <<<...>>>) is not supported: "
                               "declare __shared__ arrays with a static size");
            ok = false;
        }
    }
    for (const auto& a : callArgs) site.argExprTexts.push_back(sliceText(a.first, a.second - 1));

    i = closeParen;
    if (ok) module_.launchSites.push_back(std::move(site));
    return ok;
}

void Parser::scanLaunchSites() {
    hostScopes_.clear();
    hostScopes_.emplace_back();
    const std::size_t n = raw_.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Token& t = raw_[i];
        if (t.type == TokenType::EndOfFile) break;
        if (t.type == TokenType::LBrace) {
            hostScopes_.emplace_back();
            continue;
        }
        if (t.type == TokenType::RBrace) {
            if (hostScopes_.size() > 1) hostScopes_.pop_back();
            continue;
        }
        const bool boundary = i == 0 || raw_[i - 1].type == TokenType::Semi || raw_[i - 1].type == TokenType::LBrace ||
                              raw_[i - 1].type == TokenType::RBrace;
        if (boundary && parseHostDeclaration(i)) continue;

        if (t.type == TokenType::Identifier && i + 1 < n) {
            // Modification of a dim3 variable: d = ..., d.x = ..., d.x++
            const HostEntry* e = findHostEntry(t.text);
            if (e && e->isDim3) {
                const TokenType nx = raw_[i + 1].type;
                const bool direct = isAssignmentToken(nx);
                const bool member = nx == TokenType::Dot && i + 3 < n && raw_[i + 2].type == TokenType::Identifier &&
                                    isAssignmentToken(raw_[i + 3].type);
                if (direct || member) {
                    for (auto it = hostScopes_.rbegin(); it != hostScopes_.rend(); ++it) {
                        const auto f = it->find(t.text);
                        if (f != it->end()) {
                            f->second.modified = true;
                            break;
                        }
                    }
                }
            }
            if (raw_[i + 1].type == TokenType::TripleLess) handleLaunch(i);
        }
    }
}

} // namespace cudagfx

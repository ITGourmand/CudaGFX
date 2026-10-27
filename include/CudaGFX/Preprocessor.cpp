// Preprocessor.cpp
#include "Preprocessor.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "ConstEval.hpp"
#include "Lexer.hpp"

namespace cudagfx {

namespace {

constexpr unsigned kMaxIncludeDepth = 32;
constexpr std::size_t kMaxExpandedTokens = 2000000;

bool isIdentifierLike(const Token& t) {
    if (t.text.empty()) return false;
    const unsigned char c = static_cast<unsigned char>(t.text[0]);
    return std::isalpha(c) || c == '_' || c >= 0x80;
}

Token makeIntToken(const char* text, const Token& at) {
    Token t;
    t.type = TokenType::IntLiteral;
    t.text = text;
    t.location = at.location;
    t.length = at.length;
    return t;
}

bool readFile(const std::string& path, std::string& content) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    content = ss.str();
    return true;
}

std::string joinText(const std::vector<Token>& toks, std::size_t first, std::size_t last) {
    std::string s;
    for (std::size_t i = first; i < last; ++i) {
        if (!s.empty() && toks[i].spaceBefore) s += ' ';
        s += toks[i].text;
    }
    return s;
}

} // namespace

Preprocessor::Preprocessor(DiagnosticsEngine& diags, SourceManager& sources, std::vector<std::string> includePaths)
    : diags_(diags), sources_(sources), includePaths_(std::move(includePaths)) {
    defineMacro("__CUDAGFX__", "1");
    defineMacro("__CUDACC__", "1");
}

void Preprocessor::defineMacro(const std::string& name, const std::string& valueText) {
    if (commandLineFile_ == kNoFile) commandLineFile_ = sources_.add("<command line>", "");
    // Each definition gets its own "file" so that locations stay exact.
    const FileId file = sources_.add("<-D" + name + ">", valueText);
    DiagnosticsEngine local(&sources_);
    Lexer lexer(file, sources_.content(file), local);
    std::vector<Token> toks = lexer.tokenize();
    toks.pop_back(); // EndOfFile
    for (auto& t : toks) {
        t.atLineStart = false;
        t.location.file = commandLineFile_;
        t.location.line = 1;
        t.location.column = 1;
        t.location.offset = 0;
    }

    MacroDefinition def;
    def.name = name;
    def.body = std::move(toks);
    def.location.file = commandLineFile_;
    def.location.line = 1;
    def.location.column = 1;
    macros_[name] = std::move(def);
    for (const auto& d : local.diagnostics()) diags_.report(d.level, d.code, d.location, d.message);
}

std::string Preprocessor::resolveInclude(const std::string& header, FileId includer) const {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto exists = [&](const fs::path& p) { return fs::is_regular_file(p, ec); };

    if (sources_.isValid(includer)) {
        const std::string& from = sources_.path(includer);
        if (!from.empty() && from.front() != '<') {
            const fs::path candidate = fs::path(from).parent_path() / header;
            if (exists(candidate)) return candidate.lexically_normal().string();
        }
    }
    for (const auto& dir : includePaths_) {
        const fs::path candidate = fs::path(dir) / header;
        if (exists(candidate)) return candidate.lexically_normal().string();
    }
    return {};
}

std::vector<Token> Preprocessor::process(FileId mainFile) {
    mainFile_ = mainFile;
    mainIncludes_.clear();
    includedFiles_.clear();
    std::vector<Token> out;
    out.reserve(sources_.content(mainFile).size() / 4 + 16);
    processFile(mainFile, out, 0);
    return out;
}

std::size_t Preprocessor::directiveEnd(const std::vector<Token>& toks, std::size_t i) const {
    std::size_t j = i + 1;
    while (j < toks.size() && !toks[j].atLineStart) ++j;
    return j;
}

namespace {

// Handles the active/inactive switch and records the inactive ranges (lexical errors inside them are
// ignored, as a C preprocessor does for skipped groups).
void setActive(bool& active, std::size_t& inactiveStart,
               std::vector<std::pair<std::size_t, std::size_t>>& ranges, bool newActive, std::size_t offset) {
    if (active && !newActive) inactiveStart = offset;
    if (!active && newActive) ranges.emplace_back(inactiveStart, offset);
    active = newActive;
}

} // namespace

void Preprocessor::processFile(FileId file, std::vector<Token>& out, unsigned depth) {
    DiagnosticsEngine lexDiags(&sources_);
    Lexer lexer(file, sources_.content(file), lexDiags);

    std::vector<Token> toks = lexer.tokenize();
    std::vector<Conditional> conds;
    std::vector<std::pair<std::size_t, std::size_t>> inactiveRanges;
    bool active = true;
    std::size_t inactiveStart = 0;

    std::size_t i = 0;
    while (i < toks.size()) {
        const Token& t = toks[i];
        if (t.type == TokenType::EndOfFile) break;

        if (!(isDirective(t.type) && t.atLineStart)) {
            if (active) out.push_back(t);
            ++i;
            continue;
        }

        const std::size_t last = directiveEnd(toks, i);
        switch (t.type) {
            case TokenType::PpIf:
            case TokenType::PpIfdef:
            case TokenType::PpIfndef: {
                Conditional c;
                c.location = t.location;
                c.parentActive = active;
                bool cond = false;
                if (active) {
                    if (t.type == TokenType::PpIf) {
                        if (!evaluateCondition(toks, i + 1, last, t.location, cond)) cond = false;
                    } else if (i + 1 < last && isIdentifierLike(toks[i + 1])) {
                        const bool defined = isDefined(toks[i + 1].text);
                        cond = (t.type == TokenType::PpIfdef) ? defined : !defined;
                    } else {
                        diags_.reportError(DiagCode::CGFX0014, t.location, "Macro name expected after " + t.text);
                    }
                }
                c.taken = active ? cond : true;
                c.active = active && cond;
                conds.push_back(c);
                setActive(active, inactiveStart, inactiveRanges, c.active, t.location.offset);
                break;
            }
            case TokenType::PpElif: {
                if (conds.empty()) {
                    diags_.reportError(DiagCode::CGFX0014, t.location, "#elif without a matching #if");
                    break;
                }
                Conditional& c = conds.back();
                if (c.sawElse) diags_.reportError(DiagCode::CGFX0014, t.location, "#elif after #else");
                bool newActive = false;
                if (c.parentActive && !c.taken) {
                    bool cond = false;
                    if (!evaluateCondition(toks, i + 1, last, t.location, cond)) cond = false;
                    newActive = cond;
                    if (cond) c.taken = true;
                }
                c.active = newActive;
                setActive(active, inactiveStart, inactiveRanges, newActive, t.location.offset);
                break;
            }
            case TokenType::PpElse: {
                if (conds.empty()) {
                    diags_.reportError(DiagCode::CGFX0014, t.location, "#else without a matching #if");
                    break;
                }
                Conditional& c = conds.back();
                if (c.sawElse) diags_.reportError(DiagCode::CGFX0014, t.location, "duplicate #else");
                const bool newActive = c.parentActive && !c.taken;
                c.taken = true;
                c.sawElse = true;
                c.active = newActive;
                setActive(active, inactiveStart, inactiveRanges, newActive, t.location.offset);
                break;
            }
            case TokenType::PpEndif: {
                if (conds.empty()) {
                    diags_.reportError(DiagCode::CGFX0014, t.location, "#endif without a matching #if");
                    break;
                }
                const bool parentActive = conds.back().parentActive;
                conds.pop_back();
                setActive(active, inactiveStart, inactiveRanges, parentActive, t.location.offset);
                break;
            }
            case TokenType::PpDefine:
                if (active) handleDefine(toks, i + 1, last);
                break;
            case TokenType::PpUndef:
                if (active) {
                    if (i + 1 < last && isIdentifierLike(toks[i + 1])) macros_.erase(toks[i + 1].text);
                    else diags_.reportError(DiagCode::CGFX0014, t.location, "Macro name expected after #undef");
                }
                break;
            case TokenType::PpInclude:
                if (active) handleInclude(file, toks, i, last, out, depth);
                break;
            case TokenType::PpPragma:
                if (active && i + 1 < last && toks[i + 1].text == "once") pragmaOnce_.insert(file);
                break; // #pragma unroll & co: no effect on the analysis, removed from the stream
            case TokenType::PpLine:
                break;
            case TokenType::PpOther:
                if (!active || t.text == "#") break; // null directive
                if (t.text == "#error") {
                    diags_.reportError(DiagCode::CGFX0014, t.location, "#error: " + joinText(toks, i + 1, last));
                } else if (t.text == "#warning") {
                    diags_.reportWarning(DiagCode::CGFX0014, t.location, "#warning: " + joinText(toks, i + 1, last));
                } else {
                    diags_.reportError(DiagCode::CGFX0014, t.location,
                                       "Unsupported preprocessor directive: " + t.text);
                }
                break;
            default:
                break;
        }
        i = last;
    }

    for (const auto& c : conds) {
        diags_.reportError(DiagCode::CGFX0014, c.location, "Unterminated conditional directive (missing #endif)");
    }
    if (!active) inactiveRanges.emplace_back(inactiveStart, sources_.content(file).size() + 1);

    for (const auto& d : lexDiags.diagnostics()) {
        const std::size_t off = d.location.offset;
        const bool skipped = std::any_of(inactiveRanges.begin(), inactiveRanges.end(), [off](const auto& r) {
            return off >= r.first && off < r.second;
        });
        if (!skipped) diags_.report(d.level, d.code, d.location, d.message);
    }

    if (depth == 0) out.push_back(toks.back()); // EndOfFile of the main file
}

void Preprocessor::handleDefine(const std::vector<Token>& toks, std::size_t first, std::size_t last) {
    if (first >= last || !isIdentifierLike(toks[first])) {
        diags_.reportError(DiagCode::CGFX0014, toks[first - 1].location, "Macro name expected after #define");
        return;
    }
    MacroDefinition def;
    def.name = toks[first].text;
    def.location = toks[first].location;

    std::size_t k = first + 1;
    if (k < last && toks[k].type == TokenType::LParen && !toks[k].spaceBefore) {
        def.functionLike = true;
        ++k;
        while (true) {
            if (k >= last) {
                diags_.reportError(DiagCode::CGFX0014, toks[first].location,
                                   "Parameter list of macro '" + def.name + "' is not terminated");
                return;
            }
            if (toks[k].type == TokenType::RParen) {
                ++k;
                break;
            }
            if (toks[k].type == TokenType::Ellipsis) {
                def.variadic = true;
                ++k;
                if (k < last && toks[k].type == TokenType::RParen) {
                    ++k;
                    break;
                }
                diags_.reportError(DiagCode::CGFX0014, toks[k - 1].location, "expected ')' after '...'");
                return;
            }
            if (!isIdentifierLike(toks[k])) {
                diags_.reportError(DiagCode::CGFX0014, toks[k].location, "Invalid macro parameter '" + toks[k].text + "'");
                return;
            }
            def.params.push_back(toks[k].text);
            ++k;
            if (k < last && toks[k].type == TokenType::Comma) {
                ++k;
                continue;
            }
            if (k < last && toks[k].type == TokenType::RParen) {
                ++k;
                break;
            }
            diags_.reportError(DiagCode::CGFX0014, toks[k - 1].location, "expected ',' or ')' in the parameter list");
            return;
        }
    }

    for (; k < last; ++k) {
        const Token& b = toks[k];
        if (b.type == TokenType::Hash || b.type == TokenType::HashHash) {
            diags_.reportError(DiagCode::CGFX0014, b.location,
                               "Operator '" + b.text + "' is not supported in a macro body ('" + def.name + "')");
            return;
        }
        if (b.type == TokenType::TripleLess || b.type == TokenType::TripleGreater) {
            diags_.reportError(DiagCode::CGFX0013, b.location,
                               "Kernel launch <<<...>>> in the body of macro '" + def.name +
                                   "': not supported (write the launch directly in the host code)");
            return;
        }
        def.body.push_back(b);
    }
    if (!def.body.empty()) def.body.front().atLineStart = false;
    macros_[def.name] = std::move(def);
}

void Preprocessor::handleInclude(FileId includer, const std::vector<Token>& toks, std::size_t dirIdx,
                                 std::size_t last, std::vector<Token>& out, unsigned depth) {
    const Token& dir = toks[dirIdx];
    if (dirIdx + 1 >= last) {
        diags_.reportError(DiagCode::CGFX0014, dir.location, "Header name expected after #include");
        return;
    }
    const Token& h = toks[dirIdx + 1];
    std::string header;
    bool angled = false;
    if (h.type == TokenType::HeaderName && h.text.size() >= 2) {
        header = h.text.substr(1, h.text.size() - 2);
        angled = true;
    } else if (h.type == TokenType::StringLiteral && h.text.size() >= 2 && h.text.front() == '"') {
        header = h.text.substr(1, h.text.size() - 2);
    } else {
        diags_.reportError(DiagCode::CGFX0014, h.location, "Unsupported #include form (macro or unknown syntax)");
        return;
    }

    if (includer == mainFile_) {
        IncludeDirective inc;
        inc.span.start = dir.location;
        inc.span.end = h.location;
        inc.span.end.offset = h.endOffset();
        inc.header = header;
        inc.angled = angled;
        mainIncludes_.push_back(std::move(inc));
    }

    if (angled) return; // system headers: never followed

    const std::string path = resolveInclude(header, includer);
    if (path.empty()) return; // not found here: the host compiler will decide

    FileId id = sources_.find(path);
    if (id == kNoFile) {
        std::string content;
        if (!readFile(path, content)) return;
        id = sources_.add(path, std::move(content));
    }
    if (pragmaOnce_.count(id)) return;
    if (depth + 1 > kMaxIncludeDepth) {
        diags_.reportError(DiagCode::CGFX0014, dir.location, "Include nesting too deep (include cycle?): " + header);
        return;
    }
    if (std::find(includedFiles_.begin(), includedFiles_.end(), id) == includedFiles_.end()) includedFiles_.push_back(id);
    processFile(id, out, depth + 1);
}

bool Preprocessor::evaluateCondition(const std::vector<Token>& toks, std::size_t first, std::size_t last,
                                     const SourceLocation& at, bool& result) {
    result = false;
    std::vector<Token> expr;
    for (std::size_t k = first; k < last; ++k) {
        const Token& t = toks[k];
        if (t.type == TokenType::Identifier && t.text == "defined") {
            std::size_t n = k + 1;
            const bool paren = n < last && toks[n].type == TokenType::LParen;
            if (paren) ++n;
            if (n >= last || !isIdentifierLike(toks[n])) {
                diags_.reportError(DiagCode::CGFX0014, t.location, "Macro name expected after 'defined'");
                return false;
            }
            const bool defined = isDefined(toks[n].text);
            ++n;
            if (paren) {
                if (n >= last || toks[n].type != TokenType::RParen) {
                    diags_.reportError(DiagCode::CGFX0014, t.location, "expected ')' after defined(...)");
                    return false;
                }
                ++n;
            }
            expr.push_back(makeIntToken(defined ? "1" : "0", t));
            k = n - 1;
        } else {
            expr.push_back(t);
        }
    }
    if (expr.empty()) {
        diags_.reportError(DiagCode::CGFX0014, at, "Missing expression in the conditional directive");
        return false;
    }

    std::vector<Token> expanded = expand(expr);
    for (auto& t : expanded) {
        if (t.type == TokenType::Identifier) t = makeIntToken("0", t); // C rule: an unknown identifier is 0
    }
    ConstEvaluator evaluator;
    std::int64_t value = 0;
    if (!evaluator.evaluate(expanded.data(), expanded.data() + expanded.size(), value)) {
        diags_.reportError(DiagCode::CGFX0014, at, "Invalid conditional directive expression: " + evaluator.error());
        return false;
    }
    result = value != 0;
    return true;
}

std::vector<Token> Preprocessor::expand(const std::vector<Token>& tokens) const {
    std::vector<Token> out;
    out.reserve(tokens.size());
    std::vector<const MacroDefinition*> active;
    std::size_t budget = kMaxExpandedTokens;
    expandInto(tokens, 0, tokens.size(), out, active, budget);
    return out;
}

void Preprocessor::expandInto(const std::vector<Token>& in, std::size_t begin, std::size_t end,
                              std::vector<Token>& out, std::vector<const MacroDefinition*>& active,
                              std::size_t& budget) const {
    auto push = [&](Token t) {
        if (budget == 0) return;
        --budget;
        out.push_back(std::move(t));
    };
    auto relocate = [](Token t, const Token& invocation, unsigned length) {
        t.location = invocation.location;
        t.length = length;
        t.atLineStart = false;
        return t;
    };

    std::size_t i = begin;
    while (i < end) {
        if (budget == 0) {
            diags_.reportError(DiagCode::CGFX0014, in[i].location, "Macro expansion too large (recursion?)");
            return;
        }
        const Token& t = in[i];
        if (t.type != TokenType::Identifier) {
            push(t);
            ++i;
            continue;
        }
        const auto it = macros_.find(t.text);
        if (it == macros_.end() || std::find(active.begin(), active.end(), &it->second) != active.end()) {
            push(t);
            ++i;
            continue;
        }
        const MacroDefinition& m = it->second;

        if (!m.functionLike) {
            std::vector<Token> body;
            body.reserve(m.body.size());
            for (const auto& b : m.body) body.push_back(relocate(b, t, t.length));
            active.push_back(&m);
            expandInto(body, 0, body.size(), out, active, budget);
            active.pop_back();
            ++i;
            continue;
        }

        // Function-like macro: only invoked when '(' follows.
        const std::size_t open = i + 1;
        if (open >= end || in[open].type != TokenType::LParen) {
            push(t);
            ++i;
            continue;
        }
        std::vector<std::vector<Token>> args(1);
        int depth = 0;
        std::size_t k = open;
        bool closed = false;
        for (; k < end; ++k) {
            const Token& a = in[k];
            if (a.type == TokenType::LParen) {
                if (depth++ > 0) args.back().push_back(a);
            } else if (a.type == TokenType::RParen) {
                if (--depth == 0) {
                    closed = true;
                    break;
                }
                args.back().push_back(a);
            } else if (a.type == TokenType::Comma && depth == 1) {
                args.emplace_back();
            } else {
                args.back().push_back(a);
            }
        }
        if (!closed) {
            diags_.reportError(DiagCode::CGFX0014, t.location, "Invocation of macro '" + m.name + "' is not terminated");
            push(t);
            return;
        }
        if (m.params.empty() && !m.variadic && args.size() == 1 && args[0].empty()) args.clear();
        const bool arityOk = m.variadic ? args.size() >= m.params.size() : args.size() == m.params.size();
        if (!arityOk) {
            diags_.reportError(DiagCode::CGFX0014, t.location,
                               "Macro '" + m.name + "' expects " + std::to_string(m.params.size()) +
                                   " argument(s), " + std::to_string(args.size()) + " provided");
            push(t);
            i = k + 1;
            continue;
        }

        std::vector<std::vector<Token>> expandedArgs(args.size());
        for (std::size_t a = 0; a < args.size(); ++a) {
            expandInto(args[a], 0, args[a].size(), expandedArgs[a], active, budget);
        }
        std::vector<Token> vaArgs;
        if (m.variadic) {
            for (std::size_t a = m.params.size(); a < expandedArgs.size(); ++a) {
                if (a > m.params.size()) {
                    Token comma;
                    comma.type = TokenType::Comma;
                    comma.text = ",";
                    comma.location = t.location;
                    vaArgs.push_back(comma);
                }
                vaArgs.insert(vaArgs.end(), expandedArgs[a].begin(), expandedArgs[a].end());
            }
        }

        unsigned invocationLength = t.length;
        if (in[k].location.file == t.location.file && in[k].endOffset() > t.location.offset) {
            invocationLength = static_cast<unsigned>(in[k].endOffset() - t.location.offset);
        }

        std::vector<Token> result;
        result.reserve(m.body.size() + 8);
        for (const auto& b : m.body) {
            if (b.type == TokenType::Identifier) {
                const auto p = std::find(m.params.begin(), m.params.end(), b.text);
                if (p != m.params.end()) {
                    const auto idx = static_cast<std::size_t>(p - m.params.begin());
                    if (idx < expandedArgs.size()) {
                        result.insert(result.end(), expandedArgs[idx].begin(), expandedArgs[idx].end());
                    }
                    continue;
                }
                if (m.variadic && b.text == "__VA_ARGS__") {
                    result.insert(result.end(), vaArgs.begin(), vaArgs.end());
                    continue;
                }
            }
            result.push_back(relocate(b, t, invocationLength));
        }
        active.push_back(&m);
        expandInto(result, 0, result.size(), out, active, budget);
        active.pop_back();
        i = k + 1;
    }
}

} // namespace cudagfx

// Diagnostics.cpp
#include "Diagnostics.hpp"

#include <ostream>
#include "SourceManager.hpp"

namespace cudagfx {

void DiagnosticsEngine::report(DiagLevel level, DiagCode code, SourceLocation loc, std::string msg) {
    if (muted_) {
        if (level == DiagLevel::Error) ++mutedErrors_;
        return;
    }
    std::string key;
    key.reserve(32);
    key += std::to_string(static_cast<int>(level));
    key += ':';
    key += std::to_string(static_cast<int>(code));
    key += ':';
    key += std::to_string(loc.file);
    key += ':';
    key += std::to_string(loc.offset);
    key += ':';
    key += std::to_string(loc.line);
    if (!seen_.insert(std::move(key)).second) return;

    if (level == DiagLevel::Error) {
        if (errorCount_ >= kMaxErrors) {
            truncated_ = true;
            return;
        }
        ++errorCount_;
    } else if (level == DiagLevel::Warning) {
        ++warningCount_;
    }
    diags_.push_back(Diagnostic{level, code, loc, std::move(msg)});
}

const char* DiagnosticsEngine::codeToString(DiagCode code) {
    switch (code) {
        case DiagCode::CGFX0001: return "CGFX0001";
        case DiagCode::CGFX0002: return "CGFX0002";
        case DiagCode::CGFX0003: return "CGFX0003";
        case DiagCode::CGFX0004: return "CGFX0004";
        case DiagCode::CGFX0005: return "CGFX0005";
        case DiagCode::CGFX0006: return "CGFX0006";
        case DiagCode::CGFX0007: return "CGFX0007";
        case DiagCode::CGFX0008: return "CGFX0008";
        case DiagCode::CGFX0009: return "CGFX0009";
        case DiagCode::CGFX0010: return "CGFX0010";
        case DiagCode::CGFX0011: return "CGFX0011";
        case DiagCode::CGFX0012: return "CGFX0012";
        case DiagCode::CGFX0013: return "CGFX0013";
        case DiagCode::CGFX0014: return "CGFX0014";
        case DiagCode::CGFX0015: return "CGFX0015";
        case DiagCode::CGFX0016: return "CGFX0016";
        case DiagCode::CGFX0017: return "CGFX0017";
        case DiagCode::CGFX0018: return "CGFX0018";
        case DiagCode::CGFX0019: return "CGFX0019";
        case DiagCode::CGFX0020: return "CGFX0020";
        case DiagCode::CGFX0100: return "CGFX0100";
    }
    return "CGFX0100";
}

const char* DiagnosticsEngine::levelToString(DiagLevel level) {
    switch (level) {
        case DiagLevel::Note: return "note";
        case DiagLevel::Warning: return "warning";
        case DiagLevel::Error: return "error";
    }
    return "error";
}

namespace {

// Extracts line `line` (1-based) from `content`, without the carriage return or the trailing newline.
bool extractLine(const std::string& content, unsigned line, std::string& out) {
    std::size_t pos = 0;
    for (unsigned current = 1; current < line; ++current) {
        pos = content.find('\n', pos);
        if (pos == std::string::npos) return false;
        ++pos;
    }
    if (pos > content.size()) return false;
    std::size_t end = content.find('\n', pos);
    if (end == std::string::npos) end = content.size();
    out.assign(content, pos, end - pos);
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return true;
}

} // namespace

void DiagnosticsEngine::printAll(std::ostream& os) const {
    for (const auto& d : diags_) {
        std::string path = "<source>";
        if (sources_ && sources_->isValid(d.location.file)) path = sources_->path(d.location.file);

        os << path;
        if (d.location.isValid()) os << ':' << d.location.line << ':' << d.location.column;
        os << ": " << levelToString(d.level) << ": " << codeToString(d.code) << ": " << d.message << '\n';

        std::string lineText;
        if (sources_ && d.location.isValid() && sources_->isValid(d.location.file) &&
            extractLine(sources_->content(d.location.file), d.location.line, lineText)) {
            os << "    " << lineText << '\n';
            os << "    ";
            // Tabs are copied so that the caret stays aligned.
            for (unsigned i = 1; i < d.location.column && i <= lineText.size(); ++i) {
                os << (lineText[i - 1] == '\t' ? '\t' : ' ');
            }
            os << "^\n";
        }
    }
    if (truncated_) {
        os << "note: too many errors (" << kMaxErrors << "), the following ones are omitted\n";
    }
}

} // namespace cudagfx

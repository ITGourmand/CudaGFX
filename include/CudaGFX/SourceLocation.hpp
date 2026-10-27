// SourceLocation.hpp: source positions (lightweight and free to copy).
//
// A location does NOT carry the file name but a FileId (an index into the SourceManager):
// a Token or an AST node can therefore be copied without any allocation.
#pragma once

#include <cstddef>
#include <cstdint>

namespace cudagfx {

using FileId = std::uint32_t;
inline constexpr FileId kNoFile = 0;

struct SourceLocation {
    FileId file = kNoFile;
    unsigned line = 0;      // 1-based, 0 = invalid location
    unsigned column = 0;    // 1-based
    std::size_t offset = 0; // byte offset in the file

    bool isValid() const { return line != 0; }
};

struct SourceSpan {
    SourceLocation start;
    SourceLocation end; // exclusive
};

} // namespace cudagfx

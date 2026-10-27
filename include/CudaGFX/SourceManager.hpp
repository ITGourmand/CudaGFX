// SourceManager.hpp: owns the content of every source file that has been read (main file + #include).
#pragma once

#include <string>
#include <vector>
#include "SourceLocation.hpp"

namespace cudagfx {

class SourceManager {
public:
    // Registers a file and returns its identifier (always != kNoFile).
    FileId add(std::string path, std::string content);

    // Returns the identifier of a file already registered under this exact path, or kNoFile.
    FileId find(const std::string& path) const;

    bool isValid(FileId id) const { return id != kNoFile && id <= files_.size(); }
    const std::string& path(FileId id) const;
    const std::string& content(FileId id) const;
    std::size_t fileCount() const { return files_.size(); }

private:
    struct File {
        std::string path;
        std::string content;
    };
    std::vector<File> files_;
};

} // namespace cudagfx

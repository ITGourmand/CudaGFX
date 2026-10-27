// SourceManager.cpp
#include "SourceManager.hpp"

namespace cudagfx {

namespace {
const std::string kEmpty;
}

FileId SourceManager::add(std::string path, std::string content) {
    files_.push_back(File{std::move(path), std::move(content)});
    return static_cast<FileId>(files_.size());
}

FileId SourceManager::find(const std::string& path) const {
    for (std::size_t i = 0; i < files_.size(); ++i) {
        if (files_[i].path == path) return static_cast<FileId>(i + 1);
    }
    return kNoFile;
}

const std::string& SourceManager::path(FileId id) const {
    return isValid(id) ? files_[id - 1].path : kEmpty;
}

const std::string& SourceManager::content(FileId id) const {
    return isValid(id) ? files_[id - 1].content : kEmpty;
}

} // namespace cudagfx

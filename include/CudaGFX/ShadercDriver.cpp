// ShadercDriver.cpp
#include "ShadercDriver.hpp"

#include <array>
#include <cstdio>
#include <sstream>

#ifdef _WIN32
#define CGFX_POPEN _popen
#define CGFX_PCLOSE _pclose
#else
#include <sys/wait.h>
#define CGFX_POPEN popen
#define CGFX_PCLOSE pclose
#endif

namespace cudagfx {

ShadercDriver::ShadercDriver(std::string shadercPath, std::vector<std::string> includeDirs)
    : shadercPath_(std::move(shadercPath)), includeDirs_(std::move(includeDirs)) {}

std::string ShadercDriver::quoteArgument(const std::string& arg) {
#ifdef _WIN32
    // cmd.exe: double quotes; inner quotes are escaped, trailing backslashes are doubled.
    std::string out = "\"";
    std::size_t backslashes = 0;
    for (char c : arg) {
        if (c == '\\') {
            ++backslashes;
            out += c;
            continue;
        }
        if (c == '"') out.append(backslashes + 1, '\\');
        backslashes = 0;
        out += c;
    }
    out.append(backslashes, '\\');
    out += '"';
    return out;
#else
    // POSIX: single quotes; an inner single quote becomes '\''
    std::string out = "'";
    for (char c : arg) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += '\'';
    return out;
#endif
}

bool ShadercDriver::compile(const std::string& inputScPath, const std::string& outputPath, const ShaderTarget& target,
                            const std::string& bin2cName, std::string& error) const {
    std::ostringstream cmd;
    cmd << quoteArgument(shadercPath_) << " -f " << quoteArgument(inputScPath) << " -o " << quoteArgument(outputPath)
        << " --type compute --platform " << quoteArgument(target.platform) << " --profile " << quoteArgument(target.profile);
    for (const auto& inc : includeDirs_) cmd << " -i " << quoteArgument(inc);
    if (!bin2cName.empty()) cmd << " --bin2c " << quoteArgument(bin2cName);
    cmd << " 2>&1";

    FILE* pipe = CGFX_POPEN(cmd.str().c_str(), "r");
    if (!pipe) {
        error = "Unable to run shaderc: " + shadercPath_;
        return false;
    }
    std::array<char, 512> buffer;
    std::string output;
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) output += buffer.data();
    const int status = CGFX_PCLOSE(pipe);

    int code = status;
#ifndef _WIN32
    if (status != -1 && WIFEXITED(status)) code = WEXITSTATUS(status);
    else if (status != -1 && WIFSIGNALED(status)) code = 128 + WTERMSIG(status);
#endif
    if (code != 0) {
        error = "shaderc failed (code " + std::to_string(code) + ") for " + inputScPath + " [" + target.platform + ":" +
                target.profile + "]:\n" + output;
        if (code == 127) error += "\n(shaderc not found: check --shaderc " + shadercPath_ + ")";
        return false;
    }
    return true;
}

} // namespace cudagfx

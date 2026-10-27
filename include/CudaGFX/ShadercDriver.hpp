// ShadercDriver.hpp: running shaderc (compiling a .sc for one target).
#pragma once

#include <string>
#include <vector>
#include "ShaderTargets.hpp"

namespace cudagfx {

class ShadercDriver {
public:
    ShadercDriver(std::string shadercPath, std::vector<std::string> includeDirs);

    // Compiles `inputScPath` for `target`. If `bin2cName` is not empty, shaderc writes a C array of that
    // name (a .h header) instead of a raw binary. On failure, `error` holds the output of shaderc.
    bool compile(const std::string& inputScPath, const std::string& outputPath, const ShaderTarget& target,
                 const std::string& bin2cName, std::string& error) const;

    // Exposed for tests: quotes an argument for the platform shell.
    static std::string quoteArgument(const std::string& arg);

private:
    std::string shadercPath_;
    std::vector<std::string> includeDirs_;
};

} // namespace cudagfx

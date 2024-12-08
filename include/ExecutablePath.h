#pragma once

#include <filesystem>
#include <string>

namespace gait {

class ExecutablePath {
public:
    // Get the path to the current executable
    static std::filesystem::path getExecutablePath();
    
    // Get the path to the config file
    static std::filesystem::path getConfigPath();

private:
    // Helper functions
    static bool isProjectRoot(const std::filesystem::path& path);
    static std::filesystem::path findProjectRoot();
};

} // namespace gait
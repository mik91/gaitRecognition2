#include "ExecutablePath.h"
#include <stdexcept>
#include <iostream>
#include <fstream>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <libloaderapi.h>
#endif

namespace gait {

std::filesystem::path ExecutablePath::getExecutablePath() {
    #ifdef _WIN32
        // Windows-specific: Get executable path
        TCHAR path[MAX_PATH];
        if (GetModuleFileName(NULL, path, MAX_PATH) == 0) {
            throw std::runtime_error("Failed to get executable path");
        }
        return std::filesystem::path(path);
    #else
        // Linux-specific: Use /proc/self/exe
        return std::filesystem::canonical("/proc/self/exe");
    #endif
}

bool ExecutablePath::isProjectRoot(const std::filesystem::path& path) {
    // Check if this directory has the expected project structure
    return std::filesystem::exists(path / "config" / "paths.conf") &&
           std::filesystem::exists(path / "data");
}

std::filesystem::path ExecutablePath::findProjectRoot() {
    // Start from current directory
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path originalPath = currentPath;
    
    std::cout << "Starting search from: " << currentPath << std::endl;
    
    // Try going up the directory tree until we find the project root or hit the root directory
    while (!isProjectRoot(currentPath) && currentPath.has_parent_path()) {
        std::cout << "Checking: " << currentPath << std::endl;
        currentPath = currentPath.parent_path();
        
        // Safety check to prevent infinite loop
        if (currentPath.empty()) {
            break;
        }
    }
    
    if (isProjectRoot(currentPath)) {
        return currentPath;
    }
    
    // If we didn't find it going up, try some common relative paths
    std::vector<std::filesystem::path> commonPaths = {
        originalPath / ".." / "..",           // Two levels up
        originalPath / "..",                  // One level up
        originalPath / ".." / ".." / "..",    // Three levels up
    };
    
    for (const auto& path : commonPaths) {
        std::cout << "Trying path: " << path << std::endl;
        if (isProjectRoot(path)) {
            return path;
        }
    }
    
    throw std::runtime_error("Could not find project root directory");
}

std::filesystem::path ExecutablePath::getConfigPath() {
    try {
        // Find project root by looking for known project structure
        std::filesystem::path projectRoot = findProjectRoot();
        std::filesystem::path configPath = projectRoot / "config" / "paths.conf";
        
        std::cout << "Project root found at: " << projectRoot << std::endl;
        std::cout << "Config path: " << configPath << std::endl;
        
        if (!std::filesystem::exists(configPath)) {
            throw std::runtime_error("Config file not found at: " + configPath.string());
        }
        
        return configPath;
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Failed to determine config path: " + std::string(e.what()));
    }
}

} // namespace gait
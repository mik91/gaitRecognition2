#include "PathConfig.h"
#include <iostream>
#include <fstream>
#include <filesystem>

namespace gait {

bool PathConfig::loadConfig(const std::string& configFile) {
    configFile_ = configFile;
    paths_.clear();
    
    // Get and print current working directory for debugging
    std::cout << "Current working directory: " 
              << std::filesystem::current_path() << std::endl;
    
    std::cout << "Attempting to load config file: " << configFile << std::endl;
    
    std::ifstream file(configFile);
    if (!file.is_open()) {
        std::cerr << "Could not open config file: " << configFile << std::endl;
        std::cerr << "Full path attempted: " 
                  << std::filesystem::absolute(configFile) << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Find the separator
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            paths_[key] = value;
            std::cout << "Loaded path: " << key << " = " << value << std::endl;
        }
    }

    return true;
}

std::string PathConfig::getPath(const std::string& key) const {
    #ifdef _WIN32
        // Windows path mappings
        if (key == "DATASET_ROOT") {
            auto it = paths_.find("DATASET_ROOT");
            if (it != paths_.end()) {
                return it->second;
            }
        }

        if (key == "RESULTS_DIR") {
            auto it = paths_.find("RESULTS_DIR");
            if (it != paths_.end()) {
                return it->second;
            }
        }
    #else
        // Linux path mappings
        if (key == "DATASET_ROOT") {
            auto it = paths_.find("DATASET_LINUX_ROOT");
            if (it != paths_.end()) {
                return it->second;
            }
        }

        if (key == "RESULTS_DIR") {
            auto it = paths_.find("RESULTS_LINUX_DIR");
            if (it != paths_.end()) {
                return it->second;
            }
        }
    #endif

    // If no special mapping or not found, try direct lookup
    auto it = paths_.find(key);
    if (it != paths_.end()) {
        return it->second;
    }

    std::cerr << "Path not found for key: " << key << std::endl;
    return "";
}

void PathConfig::setPath(const std::string& key, const std::string& path) {
    paths_[key] = path;
}

bool PathConfig::savePaths() const {
    if (configFile_.empty()) {
        std::cerr << "No config file specified" << std::endl;
        return false;
    }

    std::ofstream file(configFile_);
    if (!file.is_open()) {
        std::cerr << "Could not open config file for writing: " << configFile_ << std::endl;
        return false;
    }

    file << "# Gait Recognition Path Configuration\n\n";
    for (const auto& [key, value] : paths_) {
        file << key << " = " << value << "\n";
    }

    return true;
}

} // namespace gait
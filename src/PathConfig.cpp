#include "PathConfig.h"
#include <iostream>
#include <filesystem>

namespace gait {

bool PathConfig::loadConfig(const std::string& configFile) {
    paths_.clear();
    
    #ifdef _WIN32
        // Windows paths
        paths_["DATASET_ROOT"] = "C:\\Users\\kamar\\OneDrive\\Documents\\UDEM\\Session 1\\IFT6150\\gaitRecognition2\\data\\CASIA_B";
        paths_["RESULTS_DIR"] = "C:\\Users\\kamar\\OneDrive\\Documents\\UDEM\\Session 1\\IFT6150\\gaitRecognition2\\results";
    #else
        // Linux paths
        paths_["DATASET_ROOT"] = "/u/kamarami/Documents/linux-gaitanalyzer/data/CASIA_B";
        paths_["RESULTS_DIR"] = "/u/kamarami/Documents/linux-gaitanalyzer/results";
    #endif

    try {
        std::filesystem::create_directories(paths_["RESULTS_DIR"]);
    } catch (const std::exception& e) {
        std::cerr << "Error creating results directory: " << e.what() << std::endl;
    }

    for (const auto& [key, value] : paths_) {
        std::cout << "Using path: " << key << " = " << value << std::endl;
    }

    return true;
}

std::string PathConfig::getPath(const std::string& key) const {
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
    return true;
}

} // namespace gait
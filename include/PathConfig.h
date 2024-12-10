#pragma once
#include <string>
#include <map>
#include <fstream>

namespace gait {

class PathConfig {
public:
    static PathConfig& getInstance() {
        static PathConfig instance;
        return instance;
    }

    bool loadConfig(const std::string& configFile);
    std::string getPath(const std::string& key) const;
    void setPath(const std::string& key, const std::string& path);
    bool savePaths() const;

private:
    PathConfig() = default;    
    std::map<std::string, std::string> paths_;
    std::string configFile_;
};

} // namespace gait
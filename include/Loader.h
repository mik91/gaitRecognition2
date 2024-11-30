#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

// Filesystem include and namespace handling for cross-platform compatibility
#if __has_include(<filesystem>)
    #include <filesystem>
    namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem>
    namespace fs = std::experimental::filesystem;
#else
    #error "No filesystem support available"
#endif

namespace gait {

class Loader {
public:
    // Changed constructor to take path explicitly rather than rely on CMake
    explicit Loader(const std::string& datasetPath);
    
    // Rest of the class remains the same
    std::vector<cv::Mat> loadSequence(
        const std::string& subjectId,
        const std::string& condition,
        int sequenceNumber);
    
    std::map<std::string, std::vector<std::vector<cv::Mat>>> 
    loadSubject(const std::string& subjectId);
    
    std::vector<std::string> getSubjectIds() const { return subjectIds_; }
    std::vector<std::string> getConditions() const { return conditions_; }
    std::vector<int> getSequenceNumbers(const std::string& subjectId, 
                                      const std::string& condition) const;
    
private:
    std::string datasetPath_;
    std::vector<std::string> subjectIds_;
    std::vector<std::string> conditions_;
    
    void scanDataset();
    std::string formatNumber(int number, int width) const;
};

} // namespace gait
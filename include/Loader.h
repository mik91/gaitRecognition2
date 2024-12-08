#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

namespace gait {

class Loader {
public:
    Loader(const std::string& datasetPath);
    
    // Load a sequence of frames for a given subject and condition
    std::vector<cv::Mat> loadSequence(const std::string& subjectId,
                                    const std::string& condition,
                                    int sequenceNumber);

    // Scan dataset and verify structure
    void scanDataset();

private:
    // Get the file prefix for a specific subject
    std::string getSubjectPrefix(const std::string& subjectId, 
                               const std::string& condition,
                               int sequenceNumber);
    
    // Format numbers with leading zeros
    std::string formatNumber(int number, int width) const;

    std::string datasetPath_;
    std::vector<std::string> subjectIds_;
    std::vector<std::string> conditions_;
};

} // namespace gait
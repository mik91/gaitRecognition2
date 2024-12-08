#include "Loader.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <regex>

namespace gait {

Loader::Loader(const std::string& datasetPath) 
    : datasetPath_(datasetPath) {
    scanDataset();
}

void Loader::scanDataset() {
    // Standard conditions
    conditions_ = {"bg", "cl", "nm"};
    
    // Scan for subjects
    try {
        for (const auto& entry : fs::directory_iterator(datasetPath_)) {
            if (fs::is_directory(entry)) {
                subjectIds_.push_back(entry.path().filename().string());
            }
        }
        
        std::cout << "Found " << subjectIds_.size() << " subjects in dataset" << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error scanning dataset: " << e.what() << std::endl;
    }
}

std::vector<cv::Mat> Loader::loadSequence(const std::string& subjectId,
                                        const std::string& condition,
                                        int sequenceNumber) {
    std::vector<cv::Mat> frames;
    std::string sequencePath = condition + "-" + formatNumber(sequenceNumber, 2);
    std::filesystem::path fullPath = std::filesystem::path(datasetPath_) / subjectId / sequencePath;
    
    std::cout << "Checking sequence path: " << fullPath << std::endl;
    
    if (!fs::exists(fullPath)) {
        std::cout << "Warning: Sequence path does not exist: " << fullPath << std::endl;
        return frames;
    }

    // Get prefix for this subject from actual files
    std::string prefix = getSubjectPrefix(subjectId, condition, sequenceNumber);
    if (prefix.empty()) {
        std::cout << "Warning: Could not determine prefix for subject " << subjectId << std::endl;
        return frames;
    }
    
    // Load frames
    for (int frameNum = 0; frameNum <= 180; frameNum += 18) {
        fs::path frameDir = fullPath / formatNumber(frameNum, 3);
        
        std::string framePrefix = prefix + "-" + condition + "-" + 
                                formatNumber(sequenceNumber, 2) + "-" +
                                formatNumber(frameNum, 3);
                           
        std::cout << "Looking for frames with prefix: " << framePrefix 
                 << " in " << frameDir << std::endl;
        
        if (fs::exists(frameDir) && fs::is_directory(frameDir)) {
            for (const auto& entry : fs::directory_iterator(frameDir)) {
                if (entry.path().extension() == ".png" && 
                    entry.path().filename().string().find(framePrefix) == 0) {
                    std::cout << "Loading frame: " << entry.path() << std::endl;
                    cv::Mat frame = cv::imread(entry.path().string());
                    if (!frame.empty()) {
                        frames.push_back(frame);
                        break;  // Take the first matching frame
                    }
                }
            }
        }
    }
    
    std::cout << "Loaded " << frames.size() << " frames for " 
              << subjectId << " sequence " << condition << std::endl;
    
    return frames;
}

std::string Loader::getSubjectPrefix(const std::string& subjectId, 
                                   const std::string& condition,
                                   int sequenceNumber) {
    // Build path to first frame directory
    std::string sequencePath = condition + "-" + formatNumber(sequenceNumber, 2);
    fs::path firstFrameDir = fs::path(datasetPath_) / subjectId / sequencePath / "000";
    
    if (!fs::exists(firstFrameDir) || !fs::is_directory(firstFrameDir)) {
        return "";
    }

    // Find first PNG file that matches the pattern XXX-condition-NN-000*.png
    for (const auto& entry : fs::directory_iterator(firstFrameDir)) {
        if (entry.path().extension() == ".png") {
            std::string filename = entry.path().filename().string();
            // Extract first three digits before the first hyphen
            size_t firstHyphen = filename.find('-');
            if (firstHyphen != std::string::npos && firstHyphen >= 3) {
                return filename.substr(0, 3);
            }
        }
    }
    return "";
}

std::string Loader::formatNumber(int number, int width) const {
    std::ostringstream ss;
    ss << std::setw(width) << std::setfill('0') << number;
    return ss.str();
}

} // namespace gait
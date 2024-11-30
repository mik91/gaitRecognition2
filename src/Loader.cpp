#include "Loader.h"
#include <sstream>
#include <iomanip>

namespace gait {

gait::Loader::Loader(const std::string& datasetPath) 
    : datasetPath_(datasetPath) {
    scanDataset();
}

void Loader::scanDataset() {
    // Predefined subject IDs
    subjectIds_ = {"test", "test2"};
    
    // Standard conditions
    conditions_ = {"bg", "cl", "nm"};
    
    // Verify directories exist
    for (const auto& subject : subjectIds_) {
        fs::path subjectPath = fs::path(datasetPath_) / subject;
        if (!fs::exists(subjectPath)) {
            throw std::runtime_error("Subject directory not found: " + subjectPath.string());
        }
    }
}

std::vector<cv::Mat> Loader::loadSequence(
    const std::string& subjectId,
    const std::string& condition,
    int sequenceNumber) {
    
    std::vector<cv::Mat> frames;
    std::string sequencePath = condition + "-" + formatNumber(sequenceNumber, 2);
    fs::path fullPath = fs::path(datasetPath_) / subjectId / sequencePath;
    
    if (!fs::exists(fullPath)) {
        throw std::runtime_error("Sequence path does not exist: " + fullPath.string());
    }
    
    // Load frames
    for (int frameNum = 0; frameNum <= 180; frameNum += 18) {
        std::string framePath = (fullPath / formatNumber(frameNum, 3)).string() + ".png";
        
        if (fs::exists(framePath)) {
            cv::Mat frame = cv::imread(framePath);
            if (!frame.empty()) {
                frames.push_back(frame);
            }
        }
    }
    
    return frames;
}

std::map<std::string, std::vector<std::vector<cv::Mat>>> 
Loader::loadSubject(const std::string& subjectId) {
    std::map<std::string, std::vector<std::vector<cv::Mat>>> sequences;
    
    for (const auto& condition : conditions_) {
        std::vector<int> seqNumbers = getSequenceNumbers(subjectId, condition);
        for (int seqNum : seqNumbers) {
            try {
                auto sequence = loadSequence(subjectId, condition, seqNum);
                if (!sequence.empty()) {
                    sequences[condition].push_back(sequence);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error loading sequence: " << e.what() << std::endl;
            }
        }
    }
    
    return sequences;
}

std::vector<int> Loader::getSequenceNumbers(
    const std::string& subjectId,
    const std::string& condition) const {
    
    std::vector<int> sequences;
    fs::path subjectPath = fs::path(datasetPath_) / subjectId;
    
    if (!fs::exists(subjectPath)) {
        return sequences;
    }
    
    for (int seq = 1; seq <= 6; ++seq) {
        std::string seqDir = condition + "-" + formatNumber(seq, 2);
        if (fs::exists(subjectPath / seqDir)) {
            sequences.push_back(seq);
        }
    }
    
    return sequences;
}

std::string Loader::formatNumber(int number, int width) const {
    std::ostringstream ss;
    ss << std::setw(width) << std::setfill('0') << number;
    return ss.str();
}

} // namespace gait
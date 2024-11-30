#include "Loader.h"
#include <sstream>
#include <iomanip>
#include <regex>

namespace gait {

Loader::Loader(const std::string& datasetPath) 
    : datasetPath_(datasetPath) {
    scanDataset();
}

void Loader::scanDataset() {
    // Predefined subject IDs
    subjectIds_ = {"test", "test2"};
    
    // Standard conditions
    conditions_ = {"bg", "cl", "nm"};
    
    // Verify directories exist and scan for actual structure
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
    
    std::cout << "Checking sequence path: " << fullPath << std::endl;
    
    if (!fs::exists(fullPath)) {
        throw std::runtime_error("Sequence path does not exist: " + fullPath.string());
    }
    
    // Load frames
    for (int frameNum = 0; frameNum <= 180; frameNum += 18) {
        // Create the frame directory path
        fs::path frameDir = fullPath / formatNumber(frameNum, 3);
        
        // Pattern for frame files
        std::string prefix = "001-" + condition + "-" + 
                           formatNumber(sequenceNumber, 2) + "-" +
                           formatNumber(frameNum, 3);
                           
        std::cout << "Looking for frames with prefix: " << prefix << " in " << frameDir << std::endl;
        
        if (fs::exists(frameDir) && fs::is_directory(frameDir)) {
            for (const auto& entry : fs::directory_iterator(frameDir)) {
                if (entry.path().extension() == ".png" && 
                    entry.path().filename().string().find(prefix) == 0) {
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
    
    // Regular expression to match sequence directories (e.g., "bg-01", "nm-02")
    std::regex sequencePattern(condition + "-([0-9]{2})");
    
    if (!fs::exists(subjectPath)) {
        return sequences;
    }
    
    // Scan for sequence directories
    for (const auto& entry : fs::directory_iterator(subjectPath)) {
        if (fs::is_directory(entry)) {
            std::string dirName = entry.path().filename().string();
            std::smatch matches;
            if (std::regex_match(dirName, matches, sequencePattern)) {
                if (matches.size() > 1) {
                    try {
                        int seqNum = std::stoi(matches[1]);
                        sequences.push_back(seqNum);
                    } catch (...) {
                        // Skip invalid sequence numbers
                        continue;
                    }
                }
            }
        }
    }
    
    // Sort sequence numbers
    std::sort(sequences.begin(), sequences.end());
    return sequences;
}

std::string Loader::formatNumber(int number, int width) const {
    std::ostringstream ss;
    ss << std::setw(width) << std::setfill('0') << number;
    return ss.str();
}

} // namespace gait
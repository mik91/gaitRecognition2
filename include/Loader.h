// Loader.h
#pragma once

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <future>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;

namespace gait {

class Loader {
public:
    struct SubjectData {
        std::vector<cv::Mat> frames;
        std::vector<std::string> filenames;
    };

    explicit Loader(const std::string& datasetPath);
    
    // Main loading methods
    std::pair<std::vector<cv::Mat>, std::vector<std::string>> loadSequence(
        const std::string& subjectId,
        const std::string& condition,
        int sequenceNumber);
    
    std::map<std::string, SubjectData> loadAllSubjectsWithFilenames(bool includeAllConditions = true);
    
    // Dataset management
    void scanDataset();
    
    // Getters
    const std::vector<std::string>& getSubjectIds() const { return subjectIds_; }
    const std::vector<std::string>& getConditions() const { return conditions_; }
    std::map<std::string, std::vector<cv::Mat>> loadAllSubjects(bool includeAllConditions);
    int getMaxSequenceNumber(const std::string& condition) const;
private:
    // Helper methods
    bool validateCondition(const std::string& condition);
    std::string getSubjectPrefix(const std::string& subjectId, 
                               const std::string& condition,
                               int sequenceNumber);
    std::string formatNumber(int number, int width) const;

    // Parallel processing helpers
    std::pair<std::vector<cv::Mat>, std::vector<std::string>> loadFramesParallel(
        const std::vector<fs::path>& framePaths);
    void processFrameChunk(
        const std::vector<fs::path>& paths, 
        size_t startIdx, 
        size_t endIdx,
        std::vector<cv::Mat>& outputFrames,
        std::vector<std::string>& outputFilenames);

    // Member variables
    std::string datasetPath_;
    std::vector<std::string> conditions_;
    std::vector<std::string> subjectIds_;
    std::map<std::string, int> conditionSequences_;
    const size_t threadCount_;
};

} // namespace gait
#include "Loader.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <regex>

namespace gait {

Loader::Loader(const std::string& datasetPath) 
    : datasetPath_(datasetPath),
      threadCount_(std::thread::hardware_concurrency()) {
    scanDataset();
}

bool Loader::validateCondition(const std::string& condition) {
    try {
        for (const auto& entry : fs::directory_iterator(datasetPath_)) {
            if (fs::is_directory(entry)) {
                for (int seq = 1; seq <= 6; seq++) {
                    std::string seqPath = condition + "-" + formatNumber(seq, 2);
                    if (fs::exists(entry.path() / seqPath)) {
                        return true;
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error validating condition: " << e.what() << std::endl;
    }
    return false;
}

void Loader::scanDataset() {
    conditions_.clear();
    subjectIds_.clear();
    conditionSequences_.clear();
    
    std::vector<std::string> potentialConditions = {"nm", "bg", "cl"};
    
    std::cout << "Scanning dataset path: " << datasetPath_ << std::endl;
    
    try {
        if (!fs::exists(datasetPath_)) {
            std::cerr << "Dataset path does not exist: " << datasetPath_ << std::endl;
            return;
        }

        for (const auto& condition : potentialConditions) {
            if (validateCondition(condition)) {
                conditions_.push_back(condition);
                std::cout << "Found valid condition: " << condition << std::endl;
            } else {
                std::cout << "Condition not found in dataset: " << condition << std::endl;
            }
        }

        std::mutex subjectMutex;
        std::vector<std::future<void>> futures;

        for (const auto& entry : fs::directory_iterator(datasetPath_)) {
            if (fs::is_directory(entry)) {
                futures.push_back(std::async(std::launch::async, [this, entry, &subjectMutex]() {
                    std::string subjectId = entry.path().filename().string();
                    std::map<std::string, int> localConditionSeqs;
                    
                    for (const auto& condition : conditions_) {
                        int maxSeq = 1;
                        for (int seq = 1; seq <= 6; seq++) {
                            std::string seqPath = condition + "-" + formatNumber(seq, 2);
                            if (fs::exists(entry.path() / seqPath)) {
                                maxSeq = seq;
                            }
                        }
                        localConditionSeqs[condition] = maxSeq;
                    }

                    {
                        std::lock_guard<std::mutex> lock(subjectMutex);
                        subjectIds_.push_back(subjectId);
                        for (const auto& [condition, maxSeq] : localConditionSeqs) {
                            conditionSequences_[condition] = 
                                std::max(conditionSequences_[condition], maxSeq);
                        }
                    }
                }));
            }
        }

        // Wait for all scanning to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        std::cout << "\nFound sequence counts per condition:" << std::endl;
        for (const auto& [condition, maxSeq] : conditionSequences_) {
            std::cout << condition << ": " << maxSeq << " sequences" << std::endl;
        }
        
        std::cout << "Found " << subjectIds_.size() << " subjects in dataset" << std::endl;
        
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error scanning dataset: " << e.what() << std::endl;
    }
}

std::pair<std::vector<cv::Mat>, std::vector<std::string>> Loader::loadSequence(
    const std::string& subjectId,
    const std::string& condition,
    int sequenceNumber) {
    
    auto seqIt = conditionSequences_.find(condition);
    if (seqIt == conditionSequences_.end() || sequenceNumber > seqIt->second) {
        return {{}, {}};
    }
    
    std::string seqNumberStr = formatNumber(sequenceNumber, 2);
    std::string sequencePath = condition + "-" + seqNumberStr;
    fs::path fullPath = fs::path(datasetPath_) / subjectId / sequencePath;
    
    if (!fs::exists(fullPath)) {
        return {{}, {}};
    }

    try {
        std::vector<fs::path> framePaths;
        std::string prefix = getSubjectPrefix(subjectId, condition, sequenceNumber);
        if (prefix.empty()) {
            return {{}, {}};
        }

        std::string framePrefix = prefix + "-" + condition + "-" + seqNumberStr;
        
        for (const auto& entry : fs::recursive_directory_iterator(fullPath)) {
            if (entry.is_regular_file() && 
                entry.path().extension() == ".png" &&
                entry.path().filename().string().find(framePrefix) == 0) {
                framePaths.push_back(entry.path());
            }
        }

        if (framePaths.empty()) {
            return {{}, {}};
        }

        return loadFramesParallel(framePaths);
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading sequence: " << e.what() << std::endl;
        return {{}, {}};
    }
}

std::map<std::string, Loader::SubjectData> Loader::loadAllSubjectsWithFilenames(
    bool includeAllConditions) {
    
    std::map<std::string, SubjectData> allSubjectData;
    std::mutex mapMutex;
    std::vector<std::future<void>> futures;

    for (const auto& subjectId : subjectIds_) {
        futures.push_back(std::async(std::launch::async, [this, &allSubjectData, &mapMutex, 
                                    subjectId, includeAllConditions]() {
            SubjectData subjectData;
            
            for (const auto& condition : conditions_) {
                if (!includeAllConditions && condition != "nm") {
                    continue;
                }

                int maxSeq = getMaxSequenceNumber(condition);
                for (int seq = 1; seq <= maxSeq; seq++) {
                    auto [sequenceFrames, sequenceFilenames] = 
                        loadSequence(subjectId, condition, seq);
                    
                    if (!sequenceFrames.empty()) {
                        subjectData.frames.insert(subjectData.frames.end(),
                            sequenceFrames.begin(), sequenceFrames.end());
                        subjectData.filenames.insert(subjectData.filenames.end(),
                            sequenceFilenames.begin(), sequenceFilenames.end());
                    }
                }
            }

            if (!subjectData.frames.empty()) {
                std::lock_guard<std::mutex> lock(mapMutex);
                allSubjectData[subjectId] = std::move(subjectData);
            }
        }));
    }

    for (auto& future : futures) {
        future.wait();
    }

    return allSubjectData;
}

std::string Loader::getSubjectPrefix(const std::string& subjectId, 
                                   const std::string& condition,
                                   int sequenceNumber) {
    // Build path to first frame directory
    std::string seqStr = formatNumber(sequenceNumber, 2);
    fs::path firstFrameDir = fs::path(datasetPath_) / subjectId / 
                            (condition + "-" + seqStr) / "000";
    
    
    if (!fs::exists(firstFrameDir) || !fs::is_directory(firstFrameDir)) {
        std::cout << "First frame directory does not exist: " << firstFrameDir << std::endl;
        return "";
    }

    for (const auto& entry : fs::directory_iterator(firstFrameDir)) {
        if (entry.path().extension() == ".png") {
            std::string filename = entry.path().filename().string();
            size_t firstHyphen = filename.find('-');
            if (firstHyphen != std::string::npos && firstHyphen >= 3) {
                return filename.substr(0, 3);
            }
        }
    }
    
    std::cout << "No valid prefix found in directory: " << firstFrameDir << std::endl;
    return "";
}

std::string Loader::formatNumber(int number, int width) const {
    std::ostringstream ss;
    ss << std::setw(width) << std::setfill('0') << number;
    return ss.str();
}

int Loader::getMaxSequenceNumber(const std::string& condition) const {
    auto it = conditionSequences_.find(condition);
    if (it != conditionSequences_.end()) {
        return it->second;
    }
    std::cerr << "Warning: No sequence information found for condition: " << condition 
              << ". Defaulting to 1." << std::endl;
    return 1;
}

std::map<std::string, std::vector<cv::Mat>> Loader::loadAllSubjects(bool includeAllConditions) {
    std::map<std::string, std::vector<cv::Mat>> allSubjectData;
    std::mutex mapMutex;
    std::vector<std::future<void>> futures;

    for (const auto& subjectId : subjectIds_) {
        futures.push_back(std::async(std::launch::async, [this, &allSubjectData, &mapMutex, 
                                    subjectId, includeAllConditions]() {
            std::vector<cv::Mat> subjectImages;
            
            for (const auto& condition : conditions_) {
                if (!includeAllConditions && condition != "nm") {
                    continue;
                }

                int maxSeq = getMaxSequenceNumber(condition);
                for (int seq = 1; seq <= maxSeq; seq++) {
                    auto sequenceImages = loadSequence(subjectId, condition, seq);
                    if (!sequenceImages.first.empty()) {
                        subjectImages.insert(subjectImages.end(),
                            sequenceImages.first.begin(), sequenceImages.first.end());
                    }
                }
            }

            if (!subjectImages.empty()) {
                std::lock_guard<std::mutex> lock(mapMutex);
                allSubjectData[subjectId] = std::move(subjectImages);
            }
        }));
    }

    for (auto& future : futures) {
        future.wait();
    }

    return allSubjectData;
}

std::pair<std::vector<cv::Mat>, std::vector<std::string>> Loader::loadFramesParallel(
    const std::vector<fs::path>& framePaths) {
    
    if (framePaths.empty()) {
        return {{}, {}};
    }

    size_t totalFrames = framePaths.size();
    size_t framesPerThread = totalFrames / threadCount_;
    size_t remainingFrames = totalFrames % threadCount_;

    std::vector<std::vector<cv::Mat>> threadFrames(threadCount_);
    std::vector<std::vector<std::string>> threadFilenames(threadCount_);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < threadCount_; ++i) {
        size_t startIdx = i * framesPerThread;
        size_t endIdx = (i + 1) * framesPerThread;
        if (i == threadCount_ - 1) {
            endIdx += remainingFrames;
        }

        threads.emplace_back(&Loader::processFrameChunk, this,
                           std::ref(framePaths), startIdx, endIdx,
                           std::ref(threadFrames[i]),
                           std::ref(threadFilenames[i]));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::vector<cv::Mat> allFrames;
    std::vector<std::string> allFilenames;
    size_t totalLoadedFrames = 0;
    
    for (const auto& threadFrames : threadFrames) {
        totalLoadedFrames += threadFrames.size();
    }
    
    allFrames.reserve(totalLoadedFrames);
    allFilenames.reserve(totalLoadedFrames);

    for (size_t i = 0; i < threadCount_; ++i) {
        allFrames.insert(allFrames.end(),
                        std::make_move_iterator(threadFrames[i].begin()),
                        std::make_move_iterator(threadFrames[i].end()));
        allFilenames.insert(allFilenames.end(),
                          threadFilenames[i].begin(),
                          threadFilenames[i].end());
    }

    return {allFrames, allFilenames};
}

void Loader::processFrameChunk(
    const std::vector<fs::path>& paths,
    size_t startIdx,
    size_t endIdx,
    std::vector<cv::Mat>& outputFrames,
    std::vector<std::string>& outputFilenames) {
    
    outputFrames.reserve(endIdx - startIdx);
    outputFilenames.reserve(endIdx - startIdx);
    
    for (size_t i = startIdx; i < endIdx; ++i) {
        cv::Mat frame = cv::imread(paths[i].string(), cv::IMREAD_GRAYSCALE);
        
        if (!frame.empty()) {
            cv::Mat threeChannelFrame;
            cv::cvtColor(frame, threeChannelFrame, cv::COLOR_GRAY2BGR);
            outputFrames.push_back(threeChannelFrame);
            outputFilenames.push_back(paths[i].filename().string());
        } else {
            std::cerr << "Failed to load frame: " << paths[i].string() << std::endl;
        }
    }
}
} // namespace gait
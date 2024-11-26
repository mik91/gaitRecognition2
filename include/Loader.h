// Loader.hpp
#pragma once
#include <opencv2/opencv.h>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
namespace fs = std::filesystem;

namespace gait {

struct GaitSequence {
    int subjectId;           // e.g., 001, 002
    std::string condition;   // bg, cl, or nm
    int sequenceNumber;      // e.g., 01, 02
    int frameNumber;         // e.g., 000, 018, 036
    cv::Mat frame;
    
    GaitSequence(int subject, const std::string& cond, int seq, int frame)
        : subjectId(subject), condition(cond), sequenceNumber(seq), 
          frameNumber(frame) {}
};

class Loader {
public:
    Loader(const std::string& datasetPath);
    
    // Load specific sequences
    std::vector<cv::Mat> loadSequence(int subjectId, 
                                    const std::string& condition, 
                                    int sequenceNumber);
    
    // Load all sequences for a subject
    std::map<std::string, std::vector<std::vector<cv::Mat>>> 
    loadSubject(int subjectId);
    
    // Get available subjects, conditions, and sequences
    std::vector<int> getSubjectIds() const;
    std::vector<std::string> getConditions() const;
    std::vector<int> getSequenceNumbers(int subjectId, 
                                      const std::string& condition) const;
    
    // Iterator for processing all sequences
    class SequenceIterator {
    public:
        SequenceIterator(Loader& loader);
        bool hasNext() const;
        std::vector<cv::Mat> next();
        GaitSequence getCurrentSequenceInfo() const;
        
    private:
        Loader& loader_;
        int currentSubject_;
        std::string currentCondition_;
        int currentSequence_;
        bool isValid_;
        void findNextValidSequence();
    };
    
    SequenceIterator getIterator() { return SequenceIterator(*this); }
    
private:
    std::string datasetPath_;
    std::vector<int> subjectIds_;
    std::vector<std::string> conditions_;
    
    void scanDataset();
    std::string formatNumber(int number, int width) const;
    bool isValidSequence(const fs::path& path) const;
};
    
} // namespace gait
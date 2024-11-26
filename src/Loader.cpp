#include "Loader.h"

Loader::Loader(const std::string& datasetPath) 
    : datasetPath_(datasetPath) {
    scanDataset();
}

void Loader::scanDataset() {
    // Scan for subject IDs (001, 002, etc.)
    for (const auto& entry : fs::directory_iterator(datasetPath_)) {
        if (entry.is_directory()) {
            try {
                int subjectId = std::stoi(entry.path().filename().string());
                subjectIds_.push_back(subjectId);
            } catch (...) {
                continue;
            }
        }
    }
    std::sort(subjectIds_.begin(), subjectIds_.end());
    
    // Standard conditions in CASIA Dataset
    conditions_ = {"bg", "cl", "nm"};
}

std::vector<cv::Mat> Loader::loadSequence(
    int subjectId, const std::string& condition, int sequenceNumber) {
    
    std::vector<cv::Mat> frames;
    std::string subjectPath = formatNumber(subjectId, 3);
    std::string sequencePath = condition + "-" + formatNumber(sequenceNumber, 2);
    
    fs::path fullPath = fs::path(datasetPath_) / subjectPath / sequencePath;
    
    if (!fs::exists(fullPath)) {
        throw std::runtime_error("Sequence path does not exist: " + fullPath.string());
    }
    
    // Frame numbers in CASIA are in steps of 18
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
Loader::loadSubject(int subjectId) {
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

std::string Loader::formatNumber(int number, int width) const {
    std::ostringstream ss;
    ss << std::setw(width) << std::setfill('0') << number;
    return ss.str();
}

// Iterator implementation
Loader::SequenceIterator::SequenceIterator(Loader& loader)
    : loader_(loader), currentSubject_(0), currentCondition_("bg"),
      currentSequence_(1), isValid_(true) {
    if (loader_.subjectIds_.empty()) {
        isValid_ = false;
    }
    findNextValidSequence();
}

bool Loader::SequenceIterator::hasNext() const {
    return isValid_;
}

std::vector<cv::Mat> Loader::SequenceIterator::next() {
    if (!isValid_) {
        throw std::runtime_error("No more sequences available");
    }
    
    auto sequence = loader_.loadSequence(
        loader_.subjectIds_[currentSubject_],
        currentCondition_,
        currentSequence_
    );
    
    findNextValidSequence();
    return sequence;
}

void CASIALoader::SequenceIterator::findNextValidSequence() {
    while (isValid_) {
        // Try to move to next sequence number first
        currentSequence_++;
        
        // If we've exhausted sequences for current condition, move to next condition
        if (currentSequence_ > 6) {  // CASIA has max 6 sequences per condition
            currentSequence_ = 1;
            
            // Find next condition
            if (currentCondition_ == "bg") {
                currentCondition_ = "cl";
            } else if (currentCondition_ == "cl") {
                currentCondition_ = "nm";
            } else if (currentCondition_ == "nm") {
                // If we're at nm, move to next subject
                currentCondition_ = "bg";
                currentSubject_++;
                
                // Check if we've processed all subjects
                if (currentSubject_ >= loader_.subjectIds_.size()) {
                    isValid_ = false;
                    return;
                }
            }
        }
        
        // Check if the sequence exists
        try {
            fs::path sequencePath = fs::path(loader_.datasetPath_) / 
                                  loader_.formatNumber(loader_.subjectIds_[currentSubject_], 3) /
                                  (currentCondition_ + "-" + loader_.formatNumber(currentSequence_, 2));
            
            // If the path exists and contains valid frames, we've found our next sequence
            if (fs::exists(sequencePath) && fs::is_directory(sequencePath)) {
                // Check if directory contains at least one frame
                bool hasFrames = false;
                for (const auto& entry : fs::directory_iterator(sequencePath)) {
                    if (entry.path().extension() == ".png") {
                        hasFrames = true;
                        break;
                    }
                }
                
                if (hasFrames) {
                    return;  // Valid sequence found
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error checking sequence path: " << e.what() << std::endl;
        }
        
        // If we get here, current sequence wasn't valid, continue loop to try next
    }
}

std::vector<int> CASIALoader::getSequenceNumbers(int subjectId, 
                                                const std::string& condition) const {
    std::vector<int> sequences;
    std::string subjectPath = formatNumber(subjectId, 3);
    fs::path fullPath = fs::path(datasetPath_) / subjectPath;
    
    if (!fs::exists(fullPath)) {
        return sequences;
    }
    
    for (int seq = 1; seq <= 6; ++seq) {
        std::string seqDir = condition + "-" + formatNumber(seq, 2);
        if (fs::exists(fullPath / seqDir)) {
            sequences.push_back(seq);
        }
    }
    
    return sequences;
}

std::vector<std::string> CASIALoader::getConditions() const {
    return conditions_;
}

std::vector<int> CASIALoader::getSubjectIds() const {
    return subjectIds_;
}
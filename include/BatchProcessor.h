// BatchProcessor.h
#pragma once

#include "GaitAnalyzer.h"
#include "GaitClassifier.h"
#include <vector>
#include <string>
#include <filesystem>

namespace gait {

class BatchProcessor {
public:
    struct ProcessingResult {
        std::string filename;
        std::string predictedPerson;
        double confidence;
        double processingTime;  // in milliseconds
    };

    BatchProcessor(GaitAnalyzer& analyzer, GaitClassifier& classifier) 
        : analyzer_(analyzer), classifier_(classifier) {}

    std::vector<ProcessingResult> processDirectory(
        const std::string& inputDir,
        bool visualize = false,
        const std::vector<std::string>& validExtensions = {".jpg", ".jpeg", ".png", ".bmp"});

private:
    GaitAnalyzer& analyzer_;
    GaitClassifier& classifier_;
    void summarizeResults(const std::vector<ProcessingResult>& results);
    void writeSummaryReport(const std::vector<ProcessingResult>& results, std::ofstream& file);
};

} // namespace gait
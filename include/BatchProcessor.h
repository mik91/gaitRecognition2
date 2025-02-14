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
        double processingTime;
    };

    BatchProcessor(GaitAnalyzer& analyzer, GaitClassifier& classifier) 
        : analyzer_(analyzer), classifier_(classifier) {}

    std::vector<ProcessingResult> processDirectory(
        const std::string& inputDir,
        bool visualize = false,
        const std::vector<std::string>& validExtensions = {".png"});

private:
    GaitAnalyzer& analyzer_;
    GaitClassifier& classifier_;
    void summarizeResults(const std::vector<ProcessingResult>& results);
    void writeSummaryReport(const std::vector<ProcessingResult>& results, std::ofstream& file);
    std::string extractConditionFromFilename(const std::string& filename);
};

} // namespace gait
#pragma once

#include "PersonIdentifier.h"
#include <vector>
#include <string>

namespace gait {

class BatchProcessor {
public:
    struct ProcessingResult {
        std::string filename;
        std::string predictedPerson;
        double confidence;
        double processingTime;  // in milliseconds
    };

    // Constructor
    BatchProcessor(GaitAnalyzer& analyzer, GaitClassifier& classifier);

    // Main processing method
    std::vector<ProcessingResult> processDirectory(
        const std::string& inputDir,
        bool visualize = false,
        const std::vector<std::string>& validExtensions = {".jpg", ".jpeg", ".png", ".bmp"});

private:
    PersonIdentifier identifier_;
    void summarizeResults(const std::vector<ProcessingResult>& results);
    void writeSummaryReport(const std::vector<ProcessingResult>& results, std::ofstream& file);};

} // namespace gait
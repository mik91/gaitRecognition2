#pragma once

#include "GaitAnalyzer.h"
#include "GaitClassifier.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>

namespace gait {

class PersonIdentifier {
public:
    // Constructor
    PersonIdentifier(GaitAnalyzer& analyzer, GaitClassifier& classifier);

    // Main identification methods
    std::pair<std::string, double> identifyFromImage(
        const std::string& imagePath,
        bool visualize = false);

    std::pair<std::string, double> identifyFromImage(
        const std::string& imagePath,
        bool visualize,
        const std::string& outputDir);

private:
    GaitAnalyzer& analyzer_;
    GaitClassifier& classifier_;
    
    // Helper method to save results
    void saveResults(const cv::Mat& inputImage, 
                    const cv::Mat& symmetryMap,
                    const std::string& personId,
                    double confidence,
                    const std::string& outputDir);
};

} // namespace gait
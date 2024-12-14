// PersonIdentifier.h
#pragma once

#include "GaitAnalyzer.h"
#include "GaitClassifier.h"
#include "GaitUtils.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

namespace gait {

class PersonIdentifier {
public:
    // Constructor
    PersonIdentifier(GaitAnalyzer& analyzer, GaitClassifier& classifier);

    // Updated identification methods to handle accumulated features
    std::pair<std::string, double> identifyFromImage(
        const std::string& imagePath,
        bool visualize = false);

    std::pair<std::string, double> identifyFromImage(
        const std::string& imagePath,
        bool visualize,
        const std::string& outputDir);

    // New method to identify from a sequence of images
    std::pair<std::string, double> identifyFromSequence(
        const std::vector<std::string>& imagePaths,
        bool visualize = false);

private:
    GaitAnalyzer& analyzer_;
    GaitClassifier& classifier_;
    
    // Helper methods
    void saveResults(const cv::Mat& inputImage, 
                    const cv::Mat& symmetryMap,
                    const std::string& personId,
                    double confidence,
                    const std::string& outputDir);

    std::vector<double> processImages(
        const std::vector<cv::Mat>& images,
        bool visualize,
        std::vector<cv::Mat>* symmetryMaps = nullptr);
};

} // namespace gait
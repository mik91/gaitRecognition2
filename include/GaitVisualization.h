#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace gait {
namespace visualization {

// Window management functions
bool initializeWindows();
void cleanupWindows();
bool displayResults(const cv::Mat& originalFrame, const cv::Mat& symmetryMap, 
                   const std::vector<double>& features);

// Visualization functions (moved from GaitAnalyzer)
cv::Mat visualizeSymmetryMap(const cv::Mat& symmetryMap);
cv::Mat visualizeGaitFeatures(const std::vector<double>& features);
void plotFeatureDistribution(
    const std::vector<std::vector<double>>& normalFeatures,
    const std::vector<std::vector<double>>& abnormalFeatures);

cv::Mat visualizeRegionalFeatures(const std::vector<double>& regionalFeatures);
cv::Mat visualizeTemporalFeatures(const std::vector<double>& temporalFeatures);
// cv::Mat visualizeFourierFeatures(const std::vector<double>& fourierFeatures);
} // namespace visualization
} // namespace gait
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace gait {
namespace visualization {

bool initializeWindows();
void cleanupWindows();
bool displayResults(const cv::Mat& originalFrame, const cv::Mat& symmetryMap, 
                   const std::vector<double>& features);

cv::Mat visualizeSymmetryMap(const cv::Mat& symmetryMap);
cv::Mat visualizeGaitFeatures(const std::vector<double>& features);
} // namespace visualization
} // namespace gait
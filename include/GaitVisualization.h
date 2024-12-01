#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace gait {
namespace visualization {

cv::Mat visualizeSymmetryMap(const cv::Mat& symmetryMap);
cv::Mat visualizeGaitFeatures(const std::vector<double>& features);
void plotFeatureDistribution(
    const std::vector<std::vector<double>>& normalFeatures,
    const std::vector<std::vector<double>>& abnormalFeatures);

} // namespace visualization
} // namespace gait

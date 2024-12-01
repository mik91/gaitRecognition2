#include "GaitAnalyzer.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace gait {
namespace visualization {

// Move all visualization functions here - copy from your current code
cv::Mat visualizeSymmetryMap(const cv::Mat& symmetryMap) {
    // Copy implementation from current GaitAnalyzer.cpp
}

cv::Mat visualizeGaitFeatures(const std::vector<double>& features) {
    // Copy implementation from current GaitAnalyzer.cpp
}

void plotFeatureDistribution(
    const std::vector<std::vector<double>>& normalFeatures,
    const std::vector<std::vector<double>>& abnormalFeatures) {
    // Copy implementation from current GaitAnalyzer.cpp
}

} // namespace visualization
} // namespace gait

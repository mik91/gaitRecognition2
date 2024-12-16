#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

namespace gait {

struct SymmetryParams {
    double sigma;     // Scope of symmetry detection
    double mu;        // Distance weighting
    double threshold; // Edge detection threshold
    
    SymmetryParams(double s = 27.0, double m = 90.0, double t = 0.1) 
        : sigma(s), mu(m), threshold(t) {}
};

class GaitAnalyzer {
public:
    explicit GaitAnalyzer(const SymmetryParams& params = SymmetryParams());

    cv::Mat processFrame(const cv::Mat& frame);
    std::vector<double> extractGaitFeatures(const cv::Mat& symmetryMap);

private:
    cv::Mat computeSymmetryMap(const cv::Mat& edges, const cv::Mat& gradientX, const cv::Mat& gradientY);

    SymmetryParams params_;
    cv::Mat backgroundModel_;
    bool isBackgroundInitialized_;
    std::deque<cv::Mat> backgroundFrames_;
};

} // namespace gait
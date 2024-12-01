#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <cmath>

namespace gait {

struct SymmetryParams {
    double sigma;     // Controls scope of symmetry detection
    double mu;        // Focus parameter for distance weighting
    double threshold; // Edge detection threshold
    
    SymmetryParams(double s = 27.0, double m = 90.0, double t = 0.1) 
        : sigma(s), mu(m), threshold(t) {}
};

class GaitAnalyzer {
public:
    explicit GaitAnalyzer(const SymmetryParams& params = SymmetryParams());

    // Main processing pipeline
    cv::Mat processFrame(const cv::Mat& frame);
    std::vector<double> extractGaitFeatures(const cv::Mat& symmetryMap);

private:
    // Core processing methods
    cv::Mat extractSilhouette(const cv::Mat& frame);
    std::tuple<cv::Mat, cv::Mat, cv::Mat> computeEdgesAndGradients(
        const cv::Mat& silhouette);
    cv::Mat computeSymmetryMap(const cv::Mat& edges, 
                              const cv::Mat& gradientX,
                              const cv::Mat& gradientY);
    void applyFocusWeighting(cv::Mat& symmetryMap);

    // Feature computation methods
    double computePhaseWeight(double theta1, double theta2, double alpha);
    double computeFocusWeight(const cv::Point& p1, const cv::Point& p2);
    std::vector<double> computeFourierDescriptors(const cv::Mat& symmetryMap);

    // Member variables
    SymmetryParams params_;
    cv::Mat backgroundModel_;
    bool isBackgroundInitialized_;
    std::vector<double> previousFeatures_;
};

// Optional: Utility functions for data visualization
namespace visualization {
    cv::Mat visualizeSymmetryMap(const cv::Mat& symmetryMap);
    cv::Mat visualizeGaitFeatures(const std::vector<double>& features);
    void plotFeatureDistribution(
        const std::vector<std::vector<double>>& normalFeatures,
        const std::vector<std::vector<double>>& abnormalFeatures);
}

} // namespace gait
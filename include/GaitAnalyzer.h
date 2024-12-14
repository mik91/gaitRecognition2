// GaitAnalyzer.h
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

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

    // Main analysis pipeline
    cv::Mat processFrame(const cv::Mat& frame);
    std::vector<double> extractGaitFeatures(const cv::Mat& symmetryMap);

private:
    // Core processing methods
    cv::Mat extractSilhouette(const cv::Mat& frame);
    std::tuple<cv::Mat, cv::Mat, cv::Mat> computeEdgesAndGradients(const cv::Mat& silhouette);
    cv::Mat computeSymmetryMap(const cv::Mat& edges, const cv::Mat& gradientX, const cv::Mat& gradientY);

    // Symmetry computation helpers
    double computePhaseWeight(double theta1, double theta2, double alpha);
    double computeDistanceWeight(const cv::Point& p1, const cv::Point& p2);
    double computeLogIntensity(float edgeStrength);

    // Feature extraction helpers
    std::vector<double> extractFourierDescriptors(const cv::Mat& symmetryMap);
    std::vector<double> extractLowFrequencyComponents(const cv::Mat& dft, const cv::Point& center);
    void normalizeFeatures(std::vector<double>& features);

    SymmetryParams params_;
    cv::Mat backgroundModel_;
    bool isBackgroundInitialized_;
    std::deque<cv::Mat> backgroundFrames_;
};

} // namespace gait
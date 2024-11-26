// GaitAnalyzer.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <cmath>

namespace gait {

struct SymmetryParams {
    double sigma;     // Controls scope of symmetry detection
    double mu;       // Focus parameter for distance weighting
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
    // Core symmetry computation methods
    cv::Mat computeSymmetryMap(const cv::Mat& edges, const cv::Mat& gradientX, 
                              const cv::Mat& gradientY);
    double computeSymmetryContribution(const cv::Point& p1, const cv::Point& p2,
                                     double theta1, double theta2);
    double computePhaseWeight(double theta1, double theta2, double alpha);
    double computeFocusWeight(const cv::Point& p1, const cv::Point& p2);
    
    // Preprocessing methods
    cv::Mat extractSilhouette(const cv::Mat& frame);
    std::tuple<cv::Mat, cv::Mat, cv::Mat> computeEdgesAndGradients(const cv::Mat& silhouette);
    
    // Feature extraction helpers
    std::vector<double> computeFourierDescriptors(const cv::Mat& symmetryMap);
    
    SymmetryParams params_;
    cv::Mat backgroundModel_;
    bool isBackgroundInitialized_;
};

// Utility class for managing gait cycles
class GaitCycleDetector {
public:
    GaitCycleDetector();
    bool detectCycle(const std::vector<double>& features);
    double getCyclePeriod() const;
    
private:
    std::vector<double> featureHistory_;
    double lastCyclePeriod_;
    // Additional state variables for cycle detection
};

} // namespace gait
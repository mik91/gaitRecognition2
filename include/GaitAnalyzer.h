#pragma once

#define _USE_MATH_DEFINES  // For M_PI in Windows
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <deque>

namespace gait {

struct SymmetryParams {
    double sigma;     // Controls scope of symmetry detection
    double mu;        // Focus parameter for distance weighting
    double threshold; // Edge detection threshold
    
    SymmetryParams(double s = 27.0, double m = 90.0, double t = 0.1) 
        : sigma(s), mu(m), threshold(t) {}
};

// New structure to hold different types of features
struct GaitFeatures {
    std::vector<double> regional;     // 4 regional features
    std::vector<double> temporal;     // Temporal changes
    std::vector<double> fourier;      // Fourier descriptors
    
    // Helper method to combine all features
    std::vector<double> getAllFeatures() const {
        std::vector<double> combined;
        combined.reserve(regional.size() + temporal.size() + fourier.size());
        combined.insert(combined.end(), regional.begin(), regional.end());
        combined.insert(combined.end(), temporal.begin(), temporal.end());
        combined.insert(combined.end(), fourier.begin(), fourier.end());
        return combined;
    }
};

class GaitAnalyzer {
public:
    explicit GaitAnalyzer(const SymmetryParams& params = SymmetryParams());

    // Main processing pipeline
    cv::Mat processFrame(const cv::Mat& frame);
    
    // Enhanced feature extraction methods
    GaitFeatures extractCompleteFeatures(const cv::Mat& symmetryMap);
    std::vector<double> extractGaitFeatures(const cv::Mat& symmetryMap); // For backward compatibility

private:
    // Core processing methods
    cv::Mat extractSilhouette(const cv::Mat& frame);
    std::tuple<cv::Mat, cv::Mat, cv::Mat> computeEdgesAndGradients(
        const cv::Mat& silhouette);
    cv::Mat computeSymmetryMap(const cv::Mat& edges, 
                              const cv::Mat& gradientX,
                              const cv::Mat& gradientY);
    void applyFocusWeighting(cv::Mat& symmetryMap);

    void normalizeFeatureVector(std::vector<double>& features, const std::string& name);

    // Feature computation methods
    double computePhaseWeight(double theta1, double theta2, double alpha);
    double computeFocusWeight(const cv::Point& p1, const cv::Point& p2);
    std::vector<double> computeFourierDescriptors(const cv::Mat& symmetryMap);
    
    // New feature computation methods
    std::vector<double> computeRegionalFeatures(const cv::Mat& symmetryMap);
    std::vector<double> computeTemporalFeatures(const cv::Mat& currentFeatures);

    // Member variables
    SymmetryParams params_;
    cv::Mat backgroundModel_;
    bool isBackgroundInitialized_;
    std::vector<double> previousFeatures_;
    std::deque<cv::Mat> recentMaps_;  // For temporal analysis
    static const size_t TEMPORAL_WINDOW = 5;  // Number of frames to consider for temporal analysis
};

} // namespace gait
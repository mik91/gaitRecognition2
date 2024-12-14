// GaitAnalyzer.cpp

#include "GaitAnalyzer.h"
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace gait {

GaitAnalyzer::GaitAnalyzer(const SymmetryParams& params) 
    : params_(params), isBackgroundInitialized_(false) {
}

cv::Mat GaitAnalyzer::processFrame(const cv::Mat& frame) {
    try {
        // Handle CASIA_B silhouettes which are already binary
        cv::Mat silhouette;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, silhouette, cv::COLOR_BGR2GRAY);
        } else {
            silhouette = frame.clone();
        }

        // Enhanced preprocessing
        cv::medianBlur(silhouette, silhouette, 3);  // Remove noise
        cv::threshold(silhouette, silhouette, 127, 255, cv::THRESH_BINARY);

        // Apply morphological operations for cleanup
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        cv::morphologyEx(silhouette, silhouette, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(silhouette, silhouette, cv::MORPH_OPEN, kernel);

        // Compute edges and gradients with validation
        cv::Mat edges, gradX, gradY;
        cv::Sobel(silhouette, gradX, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REFLECT);
        cv::Sobel(silhouette, gradY, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REFLECT);
        
        // Compute edge magnitude with validation
        cv::magnitude(gradX, gradY, edges);
        
        if (edges.empty() || gradX.empty() || gradY.empty()) {
            std::cerr << "Invalid gradient computation" << std::endl;
            return cv::Mat();
        }

        return computeSymmetryMap(edges, gradX, gradY);

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in processFrame: " << e.what() << std::endl;
        return cv::Mat();
    }
}

std::vector<double> GaitAnalyzer::extractGaitFeatures(const cv::Mat& symmetryMap) {
    if (symmetryMap.empty()) {
        return std::vector<double>();
    }
    
    // std::cout << "Debug - Symmetry Map stats:\n"
    //           << "Size: " << symmetryMap.size()
    //           << " Type: " << symmetryMap.type()
    //           << " Min/Max: ";
    double minVal, maxVal;
    cv::minMaxLoc(symmetryMap, &minVal, &maxVal);
    // std::cout << minVal << "/" << maxVal << std::endl;

    try {
        // Get Fourier descriptors with enhanced parameters
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(symmetryMap.rows);
        int n = cv::getOptimalDFTSize(symmetryMap.cols);
        cv::copyMakeBorder(symmetryMap, padded, 0, m - symmetryMap.rows, 0, 
                          n - symmetryMap.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        
        cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
        cv::Mat complexI;
        cv::merge(planes, 2, complexI);
        
        cv::dft(complexI, complexI);
        cv::split(complexI, planes);
        cv::magnitude(planes[0], planes[1], planes[0]);
        cv::Mat magI = planes[0];
        
        std::vector<double> features;
        features.reserve(300); // Increased reserve size for more features
        
        // Extract features in concentric circles with more detail
        cv::Point center(magI.cols/2, magI.rows/2);
        const int numCircles = 10;  // Increased from 8
        const int pointsPerCircle = 16;  // Increased from 12
        
        for (int r = 1; r <= numCircles; r++) {
            int radius = (r * std::min(center.x, center.y)) / numCircles;
            
            for (int theta = 0; theta < pointsPerCircle; theta++) {
                double angle = 2.0 * CV_PI * theta / pointsPerCircle;
                int x = center.x + radius * std::cos(angle);
                int y = center.y + radius * std::sin(angle);
                
                if (x >= 0 && x < magI.cols && y >= 0 && y < magI.rows) {
                    features.push_back(static_cast<double>(magI.at<float>(y, x)));
                }
            }
        }
        
        // Add enhanced low frequency components
        const int gridSize = 6;  // Increased from 4
        for (int y = -gridSize/2; y <= gridSize/2; y++) {
            for (int x = -gridSize/2; x <= gridSize/2; x++) {
                int sampleX = center.x + x * 4;
                int sampleY = center.y + y * 4;
                if (sampleX >= 0 && sampleX < magI.cols && 
                    sampleY >= 0 && sampleY < magI.rows) {
                    features.push_back(static_cast<double>(magI.at<float>(sampleY, sampleX)));
                }
            }
        }
        
        // Add symmetry statistics
        cv::Scalar mean, stddev;
        cv::meanStdDev(symmetryMap, mean, stddev);
        features.push_back(mean[0]);
        features.push_back(stddev[0]);
        
        // Normalize features
        normalizeFeatures(features);
        
        return features;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in feature extraction: " << e.what() << std::endl;
        return std::vector<double>();
    }
}

// In GaitAnalyzer.cpp, modify computeSymmetryMap:
cv::Mat GaitAnalyzer::computeSymmetryMap(const cv::Mat& edges, 
                                        const cv::Mat& gradientX,
                                        const cv::Mat& gradientY) {
    // Add debug info about input images
    // std::cout << "\nDebug - Symmetry Computation:\n";
    // std::cout << "Edges stats - min/max: ";
    double minVal, maxVal;
    cv::minMaxLoc(edges, &minVal, &maxVal);
    // std::cout << minVal << "/" << maxVal << std::endl;
    
    // std::cout << "GradientX stats - min/max: ";
    cv::minMaxLoc(gradientX, &minVal, &maxVal);
    // std::cout << minVal << "/" << maxVal << std::endl;
    
    // std::cout << "GradientY stats - min/max: ";
    cv::minMaxLoc(gradientY, &minVal, &maxVal);
    // std::cout << minVal << "/" << maxVal << std::endl;

    if (edges.empty() || gradientX.empty() || gradientY.empty() ||
        edges.size() != gradientX.size() || edges.size() != gradientY.size()) {
        std::cerr << "Invalid inputs to computeSymmetryMap" << std::endl;
        return cv::Mat();
    }

    cv::Mat symmetryMap = cv::Mat::zeros(edges.size(), CV_32F);
    int count = 0; // Count significant contributions
    
    // Use a smaller neighborhood for efficiency and to avoid border issues
    int searchRadius = std::min(30, std::min(edges.rows, edges.cols) / 4);
    
    // Pre-compute gradient orientations
    cv::Mat gradientAngles;
    cv::phase(gradientX, gradientY, gradientAngles);
    
    // Add debug for gradient angles
    cv::minMaxLoc(gradientAngles, &minVal, &maxVal);
    // std::cout << "Gradient angles - min/max: " << minVal << "/" << maxVal << std::endl;
    
    // Process only points with significant edge strength
    for (int y = searchRadius; y < edges.rows - searchRadius; y++) {
        for (int x = searchRadius; x < edges.cols - searchRadius; x++) {
            if (edges.at<float>(y, x) < params_.threshold) continue;
            
            float theta1 = gradientAngles.at<float>(y, x);
            float I1 = computeLogIntensity(edges.at<float>(y, x));
            
            // Search in a reduced neighborhood
            for (int dy = -searchRadius; dy <= searchRadius; dy++) {
                for (int dx = -searchRadius; dx <= searchRadius; dx++) {
                    int x2 = x + dx;
                    int y2 = y + dy;
                    
                    if (edges.at<float>(y2, x2) < params_.threshold) continue;
                    
                    float theta2 = gradientAngles.at<float>(y2, x2);
                    float I2 = computeLogIntensity(edges.at<float>(y2, x2));
                    
                    float alpha = std::atan2(static_cast<float>(dy), static_cast<float>(dx));
                    
                    float Dij = computeDistanceWeight({x, y}, {x2, y2});
                    float Phij = computePhaseWeight(theta1, theta2, alpha);
                    float contribution = Dij * Phij * I1 * I2;
                    
                    if (contribution > 1e-6) {
                        count++;
                    }
                    
                    int midX = (x + x2) / 2;
                    int midY = (y + y2) / 2;
                    symmetryMap.at<float>(midY, midX) += contribution;
                }
            }
        }
    }
    
    // std::cout << "Significant contributions: " << count << std::endl;
    
    // Normalize with validation
    cv::minMaxLoc(symmetryMap, &minVal, &maxVal);
    // std::cout << "Pre-normalization symmetry map - min/max: " << minVal << "/" << maxVal << std::endl;
    
    if (maxVal > minVal) {
        symmetryMap = (symmetryMap - minVal) / (maxVal - minVal);
    }
    
    return symmetryMap;
}

void GaitAnalyzer::normalizeFeatures(std::vector<double>& features) {
    if (features.empty()) return;
    
    // Compute robust statistics
    std::vector<double> sortedFeatures = features;
    std::sort(sortedFeatures.begin(), sortedFeatures.end());
    
    // Use median and MAD for robustness
    double median = sortedFeatures[sortedFeatures.size() / 2];
    
    std::vector<double> absDevs;
    absDevs.reserve(features.size());
    for (double f : features) {
        absDevs.push_back(std::abs(f - median));
    }
    std::sort(absDevs.begin(), absDevs.end());
    double mad = absDevs[absDevs.size() / 2] * 1.4826; // Scale factor for normal distribution
    
    // Normalize using median and MAD
    for (double& f : features) {
        f = (f - median) / (mad + 1e-10);
    }
}

double GaitAnalyzer::computePhaseWeight(double theta1, double theta2, double alpha) {
    double gradientPhase = std::abs(theta1 - theta2);
    double symmetryPhase = std::abs((theta1 + theta2 - 2 * alpha));
    return (1.0 - std::cos(gradientPhase)) * (1.0 - std::cos(symmetryPhase));
}

double GaitAnalyzer::computeDistanceWeight(const cv::Point& p1, const cv::Point& p2) {
    double distance = std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
    double normFactor = 1.0 / std::sqrt(2.0 * M_PI * params_.sigma);
    return normFactor * std::exp(-std::pow(distance - params_.mu, 2) / 
                                (2.0 * std::pow(params_.sigma, 2)));
}

double GaitAnalyzer::computeLogIntensity(float edgeStrength) {
    return std::log1p(edgeStrength);
}

} // namespace gait
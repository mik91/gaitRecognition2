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
        // (CASIA_B silhouettes are already binary)
        cv::Mat silhouette;
        if (frame.channels() == 3) {
            cv::cvtColor(frame, silhouette, cv::COLOR_BGR2GRAY);
        } else {
            silhouette = frame.clone();
        }

        cv::medianBlur(silhouette, silhouette, 3);  // Remove noise
        cv::threshold(silhouette, silhouette, 127, 255, cv::THRESH_BINARY);

        // Morphological operations (cleanup)
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
    
    std::vector<double> features;
    features.reserve(400);
    static bool firstCall = true;

    try {
        // 1. Regional Symmetry Features
        const int numVerticalRegions = 7;
        const int numHorizontalRegions = 5;
        int regionHeight = symmetryMap.rows / numVerticalRegions;
        int regionWidth = symmetryMap.cols / numHorizontalRegions;

        // Compute regional statistics
        for (int i = 0; i < numVerticalRegions; i++) {
            for (int j = 0; j < numHorizontalRegions; j++) {
                cv::Rect region(j * regionWidth, i * regionHeight, 
                              regionWidth, regionHeight);
                cv::Mat regionMat = symmetryMap(region);
                
                // Calculate statistical measures for each region
                cv::Scalar mean, stddev;
                cv::meanStdDev(regionMat, mean, stddev);
                
                // Add regional features
                features.push_back(mean[0]);     // Mean symmetry
                features.push_back(stddev[0]);   // Symmetry variation
                
                // Add max and min symmetry values
                double minVal, maxVal;
                cv::minMaxLoc(regionMat, &minVal, &maxVal);
                features.push_back(maxVal);
                features.push_back(maxVal - minVal);
            }
        }
        
        // 2. Fourier-based Features
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(symmetryMap.rows);
        int n = cv::getOptimalDFTSize(symmetryMap.cols);
        cv::copyMakeBorder(symmetryMap, padded, 0, m - symmetryMap.rows, 0, 
                          n - symmetryMap.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        
        // Prepare planes for DFT
        cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
        cv::Mat complexI;
        cv::merge(planes, 2, complexI);
        
        // Perform DFT
        cv::dft(complexI, complexI);
        cv::split(complexI, planes);
        cv::magnitude(planes[0], planes[1], planes[0]);
        cv::Mat magI = planes[0];
        
        // Log scale magnitude
        magI += cv::Scalar::all(1);
        cv::log(magI, magI);
        
        // Extract low frequency components in concentric circles
        cv::Point center(magI.cols/2, magI.rows/2);
        const int numCircles = 12;
        const int pointsPerCircle = 20;
        
        for (int r = 1; r <= numCircles; r++) {
            int radius = (r * std::min(center.x, center.y)) / numCircles;
            
            for (int theta = 0; theta < pointsPerCircle; theta++) {
                double angle = 2.0 * CV_PI * theta / pointsPerCircle;
                int x = center.x + radius * std::cos(angle);
                int y = center.y + radius * std::sin(angle);
                
                if (x >= 0 && x < magI.cols && y >= 0 && y < magI.rows) {
                    features.push_back(magI.at<float>(y, x));
                }
            }
        }
        
        // 3. Symmetry Profile Features
        // Compute vertical and horizontal symmetry profiles
        cv::Mat verticalProfile, horizontalProfile;
        cv::reduce(symmetryMap, verticalProfile, 1, cv::REDUCE_AVG);   // Vertical profile
        cv::reduce(symmetryMap, horizontalProfile, 0, cv::REDUCE_AVG); // Horizontal profile
        
        // Sample profiles at regular intervals
        const int numProfileSamples = 20;
        double vertStep = verticalProfile.rows / static_cast<double>(numProfileSamples);
        double horizStep = horizontalProfile.cols / static_cast<double>(numProfileSamples);
        
        for (int i = 0; i < numProfileSamples; i++) {
            int vIdx = static_cast<int>(i * vertStep);
            int hIdx = static_cast<int>(i * horizStep);
            
            if (vIdx < verticalProfile.rows) {
                features.push_back(verticalProfile.at<float>(vIdx));
            }
            if (hIdx < horizontalProfile.cols) {
                features.push_back(horizontalProfile.at<float>(hIdx));
            }
        }
        
        // 4. Normalize Features
        if (!features.empty()) {
            // Compute robust statistics
            std::vector<double> sortedFeatures = features;
            std::sort(sortedFeatures.begin(), sortedFeatures.end());
            
            // Use median and MAD for robust normalization
            double median = sortedFeatures[sortedFeatures.size() / 2];
            
            std::vector<double> absDevs;
            absDevs.reserve(features.size());
            for (double f : features) {
                absDevs.push_back(std::abs(f - median));
            }
            std::sort(absDevs.begin(), absDevs.end());
            // Scale factor for normal distribution
            double mad = absDevs[absDevs.size() / 2] * 1.4826; 
            
            // Normalization
            for (double& f : features) {
                f = (f - median) / (mad + 1e-10);
            }
        }

        if (firstCall) {
            std::cout << "\nFeature extraction info:" << std::endl;
            std::cout << "Number of features: " << features.size() << std::endl;
            std::cout << "Feature value range: " 
                      << *std::min_element(features.begin(), features.end()) << " to "
                      << *std::max_element(features.begin(), features.end()) << std::endl;
            firstCall = false;
        }
        
        return features;
        
    } catch (const cv::Exception& e) {
        std::cerr << "Error in feature extraction: " << e.what() << std::endl;
        return std::vector<double>();
    }
}

cv::Mat GaitAnalyzer::computeSymmetryMap(const cv::Mat& edges, 
                                        const cv::Mat& gradientX, 
                                        const cv::Mat& gradientY) {
    if (edges.empty() || gradientX.empty() || gradientY.empty() ||
        edges.size() != gradientX.size() || edges.size() != gradientY.size()) {
        std::cerr << "Invalid inputs to computeSymmetryMap" << std::endl;
        return cv::Mat();
    }

    cv::Mat symmetryMap = cv::Mat::zeros(edges.size(), CV_32F);
    
    int searchRadius = std::min(50, std::min(edges.rows, edges.cols) / 2);
    
    cv::Mat gradientAngles(edges.size(), CV_32F);
    cv::Mat logIntensities(edges.size(), CV_32F);
    
    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            gradientAngles.at<float>(y, x) = std::atan2(gradientY.at<float>(y, x),
                                                       gradientX.at<float>(y, x));
            logIntensities.at<float>(y, x) = std::log1p(edges.at<float>(y, x));
        }
    }
    
    const float edgeThreshold = 0.05f;
    
    // Process each point in the image
    #pragma omp parallel for collapse(2) 
    for (int y = searchRadius; y < edges.rows - searchRadius; y++) {
        for (int x = searchRadius; x < edges.cols - searchRadius; x++) {
            if (edges.at<float>(y, x) < edgeThreshold) continue;
            
            float theta1 = gradientAngles.at<float>(y, x);
            float I1 = logIntensities.at<float>(y, x);
            
            // Adaptive sampling based on distance
            for (int dy = -searchRadius; dy <= searchRadius; dy += 2) {
                for (int dx = -searchRadius; dx <= searchRadius; dx += 2) {
                    int x2 = x + dx;
                    int y2 = y + dy;
                    
                    if (edges.at<float>(y2, x2) < edgeThreshold) continue;
                    
                    // Compute symmetry contribution between points
                    float theta2 = gradientAngles.at<float>(y2, x2);
                    float I2 = logIntensities.at<float>(y2, x2);
                    
                    // Compute geometric relationships
                    float distance = std::sqrt(dx*dx + dy*dy);
                    float alpha = std::atan2(static_cast<float>(dy), static_cast<float>(dx));
                    
                    // Enhanced phase weighting with orientation coherence
                    float gradientPhase = std::abs(theta1 - theta2);
                    float symmetryPhase = std::abs((theta1 + theta2 - 2 * alpha));
                    float phaseWeight = (1.0f - std::cos(gradientPhase)) * 
                                      (1.0f - std::cos(symmetryPhase));
                    
                    // Distance weighting with focus parameter
                    float distanceWeight = (1.0f / std::sqrt(2.0f * M_PI * params_.sigma)) * 
                                         std::exp(-std::pow(distance - params_.mu, 2) / 
                                                (2.0f * std::pow(params_.sigma, 2)));
                    
                    // Compute final symmetry contribution
                    float contribution = distanceWeight * phaseWeight * I1 * I2;
                    
                    // Add contribution to symmetry map at midpoint
                    if (contribution > 1e-6f) {
                        int midX = (x + x2) / 2;
                        int midY = (y + y2) / 2;
                        #pragma omp atomic
                        symmetryMap.at<float>(midY, midX) += contribution;
                    }
                }
            }
        }
    }
    
    // Post-process symmetry map
    cv::GaussianBlur(symmetryMap, symmetryMap, cv::Size(3, 3), 0.5);
    
    // Normalize symmetry map to [0,1] range
    double minVal, maxVal;
    cv::minMaxLoc(symmetryMap, &minVal, &maxVal);
    
    if (maxVal > minVal) {
        symmetryMap = (symmetryMap - minVal) / (maxVal - minVal);
    }
    
    return symmetryMap;
}
} // namespace gait
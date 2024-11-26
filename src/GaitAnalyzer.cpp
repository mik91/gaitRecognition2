#include "GaitAnalyzer.h"

namespace gait {

GaitAnalyzer::GaitAnalyzer(const SymmetryParams& params) 
    : params_(params), isBackgroundInitialized_(false) {
}

cv::Mat GaitAnalyzer::processFrame(const cv::Mat& frame) {
    // Extract silhouette using background subtraction
    cv::Mat silhouette = extractSilhouette(frame);
    
    // Compute edges and gradients
    auto [edges, gradX, gradY] = computeEdgesAndGradients(silhouette);
    
    // Compute symmetry map
    return computeSymmetryMap(edges, gradX, gradY);
}

cv::Mat GaitAnalyzer::extractSilhouette(const cv::Mat& frame) {
    cv::Mat gray, silhouette;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    if (!isBackgroundInitialized_) {
        backgroundModel_ = gray.clone();
        isBackgroundInitialized_ = true;
        return cv::Mat::zeros(frame.size(), CV_8UC1);
    }
    
    // Background subtraction
    cv::absdiff(gray, backgroundModel_, silhouette);
    cv::threshold(silhouette, silhouette, params_.threshold * 255, 255, cv::THRESH_BINARY);
    
    // Morphological operations for cleanup
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(silhouette, silhouette, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(silhouette, silhouette, cv::MORPH_OPEN, kernel);
    
    return silhouette;
}

std::tuple<cv::Mat, cv::Mat, cv::Mat> GaitAnalyzer::computeEdgesAndGradients(
    const cv::Mat& silhouette) {
    
    cv::Mat gradX, gradY, edges;
    
    // Compute gradients using Sobel
    cv::Sobel(silhouette, gradX, CV_64F, 1, 0, 3);
    cv::Sobel(silhouette, gradY, CV_64F, 0, 1, 3);
    
    // Compute edge magnitude
    cv::magnitude(gradX, gradY, edges);
    
    return {edges, gradX, gradY};
}

cv::Mat GaitAnalyzer::computeSymmetryMap(const cv::Mat& edges, 
                                        const cv::Mat& gradientX,
                                        const cv::Mat& gradientY) {
    cv::Mat symmetryMap = cv::Mat::zeros(edges.size(), CV_64F);
    
    // Implementation of equations from the paper
    for (int y = 0; y < edges.rows; ++y) {
        for (int x = 0; x < edges.cols; ++x) {
            if (edges.at<double>(y, x) < params_.threshold) {
                continue;
            }
            
            // Current point angle
            double theta1 = std::atan2(gradientY.at<double>(y, x),
                                     gradientX.at<double>(y, x));
            
            // Compute symmetry contributions with nearby points
            for (int dy = -params_.sigma; dy <= params_.sigma; ++dy) {
                for (int dx = -params_.sigma; dx <= params_.sigma; ++dx) {
                    int x2 = x + dx;
                    int y2 = y + dy;
                    
                    if (x2 < 0 || x2 >= edges.cols || y2 < 0 || y2 >= edges.rows) {
                        continue;
                    }
                    
                    if (edges.at<double>(y2, x2) < params_.threshold) {
                        continue;
                    }
                    
                    double theta2 = std::atan2(gradientY.at<double>(y2, x2),
                                             gradientX.at<double>(y2, x2));
                    
                    // Compute midpoint
                    int midX = (x + x2) / 2;
                    int midY = (y + y2) / 2;
                    
                    // Add symmetry contribution to midpoint
                    symmetryMap.at<double>(midY, midX) += computeSymmetryContribution(
                        cv::Point(x, y), cv::Point(x2, y2), theta1, theta2);
                }
            }
        }
    }
    
    return symmetryMap;
}

double GaitAnalyzer::computeSymmetryContribution(const cv::Point& p1, 
                                                const cv::Point& p2,
                                                double theta1, 
                                                double theta2) {
    // Compute angle between points
    double alpha = std::atan2(p2.y - p1.y, p2.x - p1.x);
    
    // Compute phase weight (equation 6 from paper)
    double phaseWeight = computePhaseWeight(theta1, theta2, alpha);
    
    // Compute distance-based focus weight (equation 8 from paper)
    double focusWeight = computeFocusWeight(p1, p2);
    
    return phaseWeight * focusWeight;
}

std::vector<double> GaitAnalyzer::extractGaitFeatures(const cv::Mat& symmetryMap) {
    // Extract features using Fourier descriptors as described in the paper
    return computeFourierDescriptors(symmetryMap);
}

double GaitAnalyzer::computePhaseWeight(double theta1, double theta2, double alpha) {
    // Implementation of equation 6 from the paper
    // Phi(i,j) = (1 - cos(θi + θj - 2α)) × (1 - cos(θi - θj))
    
    // First factor: orientation of gradients relative to the line joining the points
    double factor1 = 1.0 - std::cos(theta1 + theta2 - 2.0 * alpha);
    
    // Second factor: relative orientation of the two gradients
    double factor2 = 1.0 - std::cos(theta1 - theta2);
    
    return factor1 * factor2;
}

double GaitAnalyzer::computeFocusWeight(const cv::Point& p1, const cv::Point& p2) {
    // Implementation of equation 8 from the paper (Focus Weighting Function)
    // FWF(i,j) = (1 / sqrt(2πσ)) * exp(-(||Pi - Pj|| - μ)^2 / (2σ^2))
    
    // Compute Euclidean distance between points
    double distance = std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
    
    // Calculate the focus weight using the parameters
    double normFactor = 1.0 / std::sqrt(2.0 * M_PI * params_.sigma);
    double expFactor = -std::pow(distance - params_.mu, 2) / (2.0 * std::pow(params_.sigma, 2));
    
    return normFactor * std::exp(expFactor);
}

std::vector<double> GaitAnalyzer::computeFourierDescriptors(const cv::Mat& symmetryMap) {
    // Convert symmetry map to suitable format for FFT
    cv::Mat complexImg;
    cv::dft(symmetryMap, complexImg, cv::DFT_COMPLEX_OUTPUT);
    
    // Split into real and imaginary parts
    std::vector<cv::Mat> planes;
    cv::split(complexImg, planes);
    cv::Mat magnitudeImg;
    cv::magnitude(planes[0], planes[1], magnitudeImg);
    
    // Apply logarithmic transform to handle different scales
    cv::Mat logMagnitude;
    cv::log(magnitudeImg + 1, logMagnitude);
    
    // Rearrange quadrants so the origin is at center
    int cx = logMagnitude.cols/2;
    int cy = logMagnitude.rows/2;
    
    cv::Mat q0(logMagnitude, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(logMagnitude, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(logMagnitude, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(logMagnitude, cv::Rect(cx, cy, cx, cy));
    
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    // Extract circular samples for rotation invariance
    std::vector<double> descriptors;
    const int numSamples = 64; // Adjustable parameter for feature vector size
    const double maxRadius = std::min(cx, cy);
    
    for (int r = 0; r < maxRadius; r += maxRadius/numSamples) {
        double sum = 0.0;
        int count = 0;
        
        // Sample points on circle of radius r
        for (double theta = 0; theta < 2*M_PI; theta += M_PI/180) {
            int x = cx + r * std::cos(theta);
            int y = cy + r * std::sin(theta);
            
            if (x >= 0 && x < logMagnitude.cols && y >= 0 && y < logMagnitude.rows) {
                sum += logMagnitude.at<float>(y, x);
                count++;
            }
        }
        
        if (count > 0) {
            descriptors.push_back(sum / count);
        }
    }
    
    // Normalize descriptors
    if (!descriptors.empty()) {
        double maxVal = *std::max_element(descriptors.begin(), descriptors.end());
        for (auto& val : descriptors) {
            val /= maxVal;
        }
    }
    
    return descriptors;
}

// Additional utility function for feature comparison
double compareGaitSignatures(const std::vector<double>& desc1, 
                           const std::vector<double>& desc2) {
    if (desc1.size() != desc2.size()) {
        throw std::runtime_error("Descriptor sizes do not match");
    }
    
    double distance = 0.0;
    for (size_t i = 0; i < desc1.size(); ++i) {
        distance += std::pow(desc1[i] - desc2[i], 2);
    }
    
    return std::sqrt(distance);
}

} // namespace gait
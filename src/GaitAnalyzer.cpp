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
    cv::Mat symmetryMap = computeSymmetryMap(edges, gradX, gradY);
    
    // Apply focus weighting
    applyFocusWeighting(symmetryMap);
    
    return symmetryMap;
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
    double otsuThresh = cv::threshold(silhouette, silhouette, 0, 255, 
                                    cv::THRESH_BINARY | cv::THRESH_OTSU);
    
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
                    
                    // Add symmetry contribution using phase weight and focus weight
                    double phaseWeight = computePhaseWeight(theta1, theta2, 
                        std::atan2(y2 - y, x2 - x));
                    double focusWeight = computeFocusWeight(
                        cv::Point(x, y), cv::Point(x2, y2));
                    
                    symmetryMap.at<double>(midY, midX) += phaseWeight * focusWeight;
                }
            }
        }
    }
    
    return symmetryMap;
}

void GaitAnalyzer::applyFocusWeighting(cv::Mat& symmetryMap) {
    cv::Mat weighted = cv::Mat::zeros(symmetryMap.size(), CV_64F);
    
    for(int y = 0; y < symmetryMap.rows; y++) {
        for(int x = 0; x < symmetryMap.cols; x++) {
            double distance = std::sqrt(x*x + y*y);
            double fwf = (1.0 / std::sqrt(2.0 * M_PI * params_.sigma)) * 
                        std::exp(-std::pow(distance - params_.mu, 2) / 
                               (2.0 * std::pow(params_.sigma, 2)));
            
            weighted.at<double>(y, x) = symmetryMap.at<double>(y, x) * fwf;
        }
    }
    symmetryMap = weighted;
}

double GaitAnalyzer::computePhaseWeight(double theta1, double theta2, double alpha) {
    // Implementation of equation 6 from the paper
    double factor1 = 1.0 - std::cos(theta1 + theta2 - 2.0 * alpha);
    double factor2 = 1.0 - std::cos(theta1 - theta2);
    return factor1 * factor2;
}

double GaitAnalyzer::computeFocusWeight(const cv::Point& p1, const cv::Point& p2) {
    double distance = std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
    double normFactor = 1.0 / std::sqrt(2.0 * M_PI * params_.sigma);
    double expFactor = -std::pow(distance - params_.mu, 2) / 
                      (2.0 * std::pow(params_.sigma, 2));
    
    return normFactor * std::exp(expFactor);
}

std::vector<double> GaitAnalyzer::extractGaitFeatures(const cv::Mat& symmetryMap) {
    std::vector<double> features;
    
    // 1. Regional analysis
    int rows = symmetryMap.rows;
    int cols = symmetryMap.cols;
    int numRegions = 4;
    
    for(int i = 0; i < numRegions; i++) {
        int startRow = (i * rows) / numRegions;
        int endRow = ((i + 1) * rows) / numRegions;
        cv::Mat region = symmetryMap(cv::Range(startRow, endRow), 
                                   cv::Range(0, cols));
        features.push_back(cv::mean(region)[0]);
    }
    
    // 2. Temporal features
    if (!previousFeatures_.empty()) {
        double featureDiff = 0.0;
        for(size_t i = 0; i < features.size(); i++) {
            featureDiff += std::abs(features[i] - previousFeatures_[i]);
        }
        features.push_back(featureDiff);
    }
    
    previousFeatures_ = features;
    
    // 3. Add Fourier descriptors
    std::vector<double> fourierFeatures = computeFourierDescriptors(symmetryMap);
    features.insert(features.end(), fourierFeatures.begin(), fourierFeatures.end());
    
    return features;
}

std::vector<double> GaitAnalyzer::computeFourierDescriptors(const cv::Mat& symmetryMap) {
    // This part can remain the same as your current implementation
    cv::Mat complexImg;
    cv::dft(symmetryMap, complexImg, cv::DFT_COMPLEX_OUTPUT);
    
    std::vector<cv::Mat> planes;
    cv::split(complexImg, planes);
    cv::Mat magnitudeImg;
    cv::magnitude(planes[0], planes[1], magnitudeImg);
    
    cv::Mat logMagnitude;
    cv::log(magnitudeImg + 1, logMagnitude);
    
    // Extract circular samples
    std::vector<double> descriptors;
    const int numSamples = 64;
    const double maxRadius = std::min(logMagnitude.rows/2, logMagnitude.cols/2);
    
    for (int r = 0; r < maxRadius; r += maxRadius/numSamples) {
        double sum = 0.0;
        int count = 0;
        
        for (double theta = 0; theta < 2*M_PI; theta += M_PI/180) {
            int x = logMagnitude.cols/2 + r * std::cos(theta);
            int y = logMagnitude.rows/2 + r * std::sin(theta);
            
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
} // namespace gait
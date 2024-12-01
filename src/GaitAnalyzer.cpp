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
namespace visualization {

cv::Mat visualizeSymmetryMap(const cv::Mat& symmetryMap) {
    cv::Mat normalized;
    cv::Mat colored;
    
    // Normalize the symmetry map to 0-1 range
    cv::normalize(symmetryMap, normalized, 0, 1, cv::NORM_MINMAX, CV_32F);
    
    // Apply colormap (COLORMAP_JET gives good visualization for symmetry)
    cv::Mat visualMap;
    normalized.convertTo(visualMap, CV_8UC1, 255);
    cv::applyColorMap(visualMap, colored, cv::COLORMAP_JET);
    
    // Add contours for better visualization of symmetry regions
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Mat> levels;
    for(double level = 0.2; level < 1.0; level += 0.2) {
        cv::Mat binary;
        cv::threshold(normalized, binary, level, 1.0, cv::THRESH_BINARY);
        binary.convertTo(binary, CV_8UC1, 255);
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(colored, contours, -1, cv::Scalar(255,255,255), 1);
    }
    
    return colored;
}

cv::Mat visualizeGaitFeatures(const std::vector<double>& features) {
    const int height = 400;
    const int width = 800;
    cv::Mat visualization = cv::Mat::zeros(height, width, CV_8UC3);
    
    // Draw background
    visualization.setTo(cv::Scalar(255, 255, 255));
    
    // Find feature range
    double minVal = *std::min_element(features.begin(), features.end());
    double maxVal = *std::max_element(features.begin(), features.end());
    double range = maxVal - minVal;
    
    // Draw grid lines
    for(int i = 0; i < 10; i++) {
        int y = i * height / 10;
        cv::line(visualization, cv::Point(0, y), cv::Point(width, y), 
                cv::Scalar(200,200,200), 1);
        
        // Add value labels
        double value = maxVal - (i * range / 10);
        cv::putText(visualization, cv::format("%.2f", value), 
                   cv::Point(5, y - 5), cv::FONT_HERSHEY_SIMPLEX, 
                   0.4, cv::Scalar(0,0,0));
    }
    
    // Draw feature bars
    int barWidth = width / features.size();
    for(size_t i = 0; i < features.size(); i++) {
        double normalizedValue = (features[i] - minVal) / range;
        int barHeight = static_cast<int>(normalizedValue * height);
        cv::Point pt1(i * barWidth, height);
        cv::Point pt2((i+1) * barWidth - 2, height - barHeight);
        
        // Use different colors for different feature types
        cv::Scalar color;
        if(i < 4) { // Regional features
            color = cv::Scalar(255, 0, 0);  // Blue
        } else if(i == 4) { // Temporal feature
            color = cv::Scalar(0, 255, 0);  // Green
        } else { // Fourier descriptors
            color = cv::Scalar(0, 0, 255);  // Red
        }
        
        cv::rectangle(visualization, pt1, pt2, color, cv::FILLED);
        cv::rectangle(visualization, pt1, pt2, cv::Scalar(0,0,0), 1);
    }
    
    // Add legend
    cv::putText(visualization, "Regional", cv::Point(10, 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0));
    cv::putText(visualization, "Temporal", cv::Point(10, 40), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0));
    cv::putText(visualization, "Fourier", cv::Point(10, 60), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255));
    
    return visualization;
}

void plotFeatureDistribution(
    const std::vector<std::vector<double>>& normalFeatures,
    const std::vector<std::vector<double>>& abnormalFeatures) {
    
    if(normalFeatures.empty() || abnormalFeatures.empty() || 
       normalFeatures[0].empty() || abnormalFeatures[0].empty()) {
        return;
    }
    
    const int height = 600;
    const int width = 800;
    cv::Mat plot = cv::Mat::zeros(height, width, CV_8UC3);
    plot.setTo(cv::Scalar(255, 255, 255));
    
    // Calculate statistics for each feature
    size_t numFeatures = normalFeatures[0].size();
    std::vector<std::pair<double, double>> normalStats(numFeatures); // mean, std
    std::vector<std::pair<double, double>> abnormalStats(numFeatures);
    
    // Calculate statistics
    for(size_t i = 0; i < numFeatures; i++) {
        // Normal stats
        double normalSum = 0, normalSqSum = 0;
        for(const auto& sample : normalFeatures) {
            normalSum += sample[i];
            normalSqSum += sample[i] * sample[i];
        }
        double normalMean = normalSum / normalFeatures.size();
        double normalStd = std::sqrt(normalSqSum/normalFeatures.size() - normalMean*normalMean);
        normalStats[i] = {normalMean, normalStd};
        
        // Abnormal stats
        double abnormalSum = 0, abnormalSqSum = 0;
        for(const auto& sample : abnormalFeatures) {
            abnormalSum += sample[i];
            abnormalSqSum += sample[i] * sample[i];
        }
        double abnormalMean = abnormalSum / abnormalFeatures.size();
        double abnormalStd = std::sqrt(abnormalSqSum/abnormalFeatures.size() - abnormalMean*abnormalMean);
        abnormalStats[i] = {abnormalMean, abnormalStd};
    }
    
    // Find global min and max for scaling
    double globalMin = std::numeric_limits<double>::max();
    double globalMax = std::numeric_limits<double>::lowest();
    
    for(size_t i = 0; i < numFeatures; i++) {
        globalMin = std::min(globalMin, normalStats[i].first - 2*normalStats[i].second);
        globalMin = std::min(globalMin, abnormalStats[i].first - 2*abnormalStats[i].second);
        globalMax = std::max(globalMax, normalStats[i].first + 2*normalStats[i].second);
        globalMax = std::max(globalMax, abnormalStats[i].first + 2*abnormalStats[i].second);
    }
    
    // Draw distribution for each feature
    int featureWidth = width / numFeatures;
    for(size_t i = 0; i < numFeatures; i++) {
        int centerX = (i + 0.5) * featureWidth;
        
        // Draw normal distribution
        auto drawDistribution = [&](const std::pair<double, double>& stats, 
                                  const cv::Scalar& color, int offset) {
            double mean = stats.first;
            double std = stats.second;
            double normalizedMean = (mean - globalMin) / (globalMax - globalMin);
            int meanY = height - static_cast<int>(normalizedMean * (height - 100));
            
            // Draw mean line
            cv::line(plot, cv::Point(centerX - 20 + offset, meanY), 
                    cv::Point(centerX + 20 + offset, meanY), color, 2);
            
            // Draw std range
            double normalizedStdTop = ((mean + std) - globalMin) / (globalMax - globalMin);
            double normalizedStdBottom = ((mean - std) - globalMin) / (globalMax - globalMin);
            int stdTopY = height - static_cast<int>(normalizedStdTop * (height - 100));
            int stdBottomY = height - static_cast<int>(normalizedStdBottom * (height - 100));
            
            cv::line(plot, cv::Point(centerX + offset, stdTopY), 
                    cv::Point(centerX + offset, stdBottomY), color, 2);
        };
        
        // Draw distributions
        drawDistribution(normalStats[i], cv::Scalar(0,255,0), -10);   // Normal in green
        drawDistribution(abnormalStats[i], cv::Scalar(0,0,255), 10);  // Abnormal in red
        
        // Add feature number
        cv::putText(plot, cv::format("F%zu", i), 
                   cv::Point(centerX - 15, height - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0));
    }
    
    // Add legend
    cv::putText(plot, "Normal", cv::Point(10, 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0));
    cv::putText(plot, "Abnormal", cv::Point(10, 40), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255));
    
    // Show plot
    cv::imshow("Feature Distribution", plot);
    cv::waitKey(1);
}

} // namespace visualization
} // namespace gait
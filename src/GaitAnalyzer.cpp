#include "GaitAnalyzer.h"

namespace gait {

GaitAnalyzer::GaitAnalyzer(const SymmetryParams& params) 
    : params_(params), isBackgroundInitialized_(false) {
}

cv::Mat GaitAnalyzer::processFrame(const cv::Mat& frame) {
    // Extract silhouette using background subtraction
    cv::Mat silhouette = extractSilhouette(frame);
    
    // Convert silhouette to float type for gradient computation
    cv::Mat silhouetteFloat;
    silhouette.convertTo(silhouetteFloat, CV_32F, 1.0/255.0);
    
    // Compute edges and gradients
    auto [edges, gradX, gradY] = computeEdgesAndGradients(silhouetteFloat);
    
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
    cv::Sobel(silhouette, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(silhouette, gradY, CV_32F, 0, 1, 3);
    
    // Compute edge magnitude
    cv::magnitude(gradX, gradY, edges);
    
    return {edges, gradX, gradY};
}

cv::Mat GaitAnalyzer::computeSymmetryMap(const cv::Mat& edges, 
                                        const cv::Mat& gradientX,
                                        const cv::Mat& gradientY) {
    cv::Mat symmetryMap = cv::Mat::zeros(edges.size(), CV_32F);
    
    // Implementation of equations from the paper
    for (int y = 0; y < edges.rows; ++y) {
        for (int x = 0; x < edges.cols; ++x) {
            if (edges.at<float>(y, x) < params_.threshold) {
                continue;
            }
            
            // Current point angle
            double theta1 = std::atan2(gradientY.at<float>(y, x),
                                     gradientX.at<float>(y, x));
            
            // Compute symmetry contributions with nearby points
            for (int dy = -params_.sigma; dy <= params_.sigma; ++dy) {
                for (int dx = -params_.sigma; dx <= params_.sigma; ++dx) {
                    int x2 = x + dx;
                    int y2 = y + dy;
                    
                    if (x2 < 0 || x2 >= edges.cols || y2 < 0 || y2 >= edges.rows) {
                        continue;
                    }
                    
                    if (edges.at<float>(y2, x2) < params_.threshold) {
                        continue;
                    }
                    
                    double theta2 = std::atan2(gradientY.at<float>(y2, x2),
                                             gradientX.at<float>(y2, x2));
                    
                    // Compute midpoint
                    int midX = (x + x2) / 2;
                    int midY = (y + y2) / 2;
                    
                    // Add symmetry contribution using phase weight and focus weight
                    double phaseWeight = computePhaseWeight(theta1, theta2, 
                        std::atan2(y2 - y, x2 - x));
                    double focusWeight = computeFocusWeight(
                        cv::Point(x, y), cv::Point(x2, y2));
                    
                    symmetryMap.at<float>(midY, midX) += 
                        static_cast<float>(phaseWeight * focusWeight);
                }
            }
        }
    }
    
    return symmetryMap;
}

void GaitAnalyzer::applyFocusWeighting(cv::Mat& symmetryMap) {
    cv::Mat weighted = cv::Mat::zeros(symmetryMap.size(), CV_32F);
    
    for(int y = 0; y < symmetryMap.rows; y++) {
        for(int x = 0; x < symmetryMap.cols; x++) {
            double distance = std::sqrt(x*x + y*y);
            double fwf = (1.0 / std::sqrt(2.0 * M_PI * params_.sigma)) * 
                        std::exp(-std::pow(distance - params_.mu, 2) / 
                               (2.0 * std::pow(params_.sigma, 2)));
            
            weighted.at<float>(y, x) = 
                symmetryMap.at<float>(y, x) * static_cast<float>(fwf);
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

std::vector<double> GaitAnalyzer::computeRegionalFeatures(const cv::Mat& symmetryMap) {
    std::vector<double> features;
    const int numRegions = 4;
    
    int rows = symmetryMap.rows;
    int cols = symmetryMap.cols;
    
    // Use larger regions to reduce sensitivity to small variations
    features.reserve(numRegions);
    for(int i = 0; i < numRegions; i++) {
        int startRow = (i * rows) / numRegions;
        int endRow = ((i + 1) * rows) / numRegions;
        cv::Mat region = symmetryMap(cv::Range(startRow, endRow), cv::Range(0, cols));
        
        // Calculate regional statistics
        cv::Scalar mean, stdDev;
        cv::meanStdDev(region, mean, stdDev);
        
        // Use both mean and standard deviation as features
        double regionFeature = mean[0];
        if (stdDev[0] > 1e-6) {  // Only add variation if it's meaningful
            regionFeature *= (1.0 + stdDev[0]);
        }
        
        features.push_back(regionFeature);
    }
    
    return features;
}

std::vector<double> GaitAnalyzer::computeTemporalFeatures(const cv::Mat& currentMap) {
    std::vector<double> temporalFeatures;
    
    // Store current map for temporal analysis
    if (recentMaps_.size() >= TEMPORAL_WINDOW) {
        recentMaps_.pop_front();
    }
    recentMaps_.push_back(currentMap.clone());

    // Need at least 2 frames for temporal features
    if (recentMaps_.size() >= 2) {
        // Compute difference between current and previous frame
        cv::Mat diff;
        cv::absdiff(recentMaps_.back(), recentMaps_[recentMaps_.size()-2], diff);
        
        // Initialize statistical measures
        cv::Scalar meanVal(0.0);
        cv::Scalar stdDevVal(0.0);
        cv::meanStdDev(diff, meanVal, stdDevVal);
        
        // Get the actual values from Scalar objects
        double meanDiff = meanVal[0];
        double stdDev = stdDevVal[0];
        
        // Create thresholded image using mean + stddev as threshold
        cv::Mat thresholded;
        double threshold = meanDiff + stdDev;
        cv::threshold(diff, thresholded, threshold, 1.0, cv::THRESH_BINARY);
        
        // Calculate area of significant change
        double changeArea = cv::sum(thresholded)[0] / (diff.rows * diff.cols);
        temporalFeatures.push_back(changeArea);
        
        // Add statistical measures
        temporalFeatures.push_back(meanDiff);
        temporalFeatures.push_back(stdDev);
        
        std::cout << "Temporal features - Change metrics:\n"
                  << "  Area of change: " << changeArea << "\n"
                  << "  Mean difference: " << meanDiff << "\n"
                  << "  StdDev of difference: " << stdDev << "\n";
    } else {
        // Fill with zeros if not enough frames
        temporalFeatures = std::vector<double>(3, 0.0);
        std::cout << "Temporal features - Not enough frames, using zeros\n";
    }
    
    return temporalFeatures;
}

GaitFeatures GaitAnalyzer::extractCompleteFeatures(const cv::Mat& symmetryMap) {
    GaitFeatures features;
    
    // Convert to float if needed
    cv::Mat floatMap;
    if (symmetryMap.type() != CV_32F) {
        symmetryMap.convertTo(floatMap, CV_32F);
    } else {
        floatMap = symmetryMap.clone();
    }
    
    // Extract regional features with validation
    features.regional = computeRegionalFeatures(floatMap);
    std::cout << "Regional features (" << features.regional.size() << "):\n";
    for (size_t i = 0; i < features.regional.size(); i++) {
        std::cout << features.regional[i] << " ";
    }
    std::cout << "\n";
    
    // Extract temporal features with validation
    features.temporal = computeTemporalFeatures(floatMap);
    std::cout << "Temporal features (" << features.temporal.size() << "):\n";
    for (size_t i = 0; i < features.temporal.size(); i++) {
        std::cout << features.temporal[i] << " ";
    }
    std::cout << "\n";
    
    // Extract Fourier features with validation
    features.fourier = computeFourierDescriptors(floatMap);
    std::cout << "Fourier features (" << features.fourier.size() << "):\n";
    for (size_t i = 0; i < std::min(size_t(10), features.fourier.size()); i++) {
        std::cout << features.fourier[i] << " ";
    }
    std::cout << "...\n";
    
    // Validate all features before combining
    auto validateFeatures = [](std::vector<double>& feats, const std::string& name) {
        for (size_t i = 0; i < feats.size(); i++) {
            if (std::isnan(feats[i]) || std::isinf(feats[i])) {
                std::cout << "Warning: Invalid " << name << " feature at index " << i 
                         << ", replacing with 0\n";
                feats[i] = 0.0;
            }
            // Handle extremely small values
            if (std::abs(feats[i]) < 1e-10) {
                feats[i] = 0.0;
            }
        }
    };
    
    validateFeatures(features.regional, "regional");
    validateFeatures(features.temporal, "temporal");
    validateFeatures(features.fourier, "Fourier");
    
    return features;
}

std::vector<double> GaitAnalyzer::extractGaitFeatures(const cv::Mat& symmetryMap) {
    GaitFeatures features;
    
    // Convert to float if needed
    cv::Mat floatMap;
    if (symmetryMap.type() != CV_32F) {
        symmetryMap.convertTo(floatMap, CV_32F);
    } else {
        floatMap = symmetryMap.clone();
    }
    
    // Extract and normalize regional features
    features.regional = computeRegionalFeatures(floatMap);
    normalizeFeatureVector(features.regional, "Regional");
    
    // Extract and normalize temporal features
    features.temporal = computeTemporalFeatures(floatMap);
    normalizeFeatureVector(features.temporal, "Temporal");
    
    // Extract and normalize Fourier features
    features.fourier = computeFourierDescriptors(floatMap);
    normalizeFeatureVector(features.fourier, "Fourier");
    
    // Combine normalized features
    std::vector<double> combinedFeatures;
    const size_t EXPECTED_FEATURE_SIZE = 124;  // Fixed size based on training data
    
    // Reserve space for the expected size
    combinedFeatures.reserve(EXPECTED_FEATURE_SIZE);
    
    // Add regional features (pad or truncate to expected size)
    size_t regionalSize = std::min(features.regional.size(), size_t(4));
    combinedFeatures.insert(combinedFeatures.end(), 
                           features.regional.begin(), 
                           features.regional.begin() + regionalSize);
    // Pad with zeros if needed
    while (combinedFeatures.size() < 4) {
        combinedFeatures.push_back(0.0);
    }
    
    // Add temporal features (pad or truncate to expected size)
    size_t temporalSize = std::min(features.temporal.size(), size_t(3));
    combinedFeatures.insert(combinedFeatures.end(), 
                           features.temporal.begin(), 
                           features.temporal.begin() + temporalSize);
    while (combinedFeatures.size() < 7) {
        combinedFeatures.push_back(0.0);
    }
    
    // Add Fourier features (pad or truncate to expected size)
    size_t fourierSize = std::min(features.fourier.size(), size_t(117));
    combinedFeatures.insert(combinedFeatures.end(), 
                           features.fourier.begin(), 
                           features.fourier.begin() + fourierSize);
    // Pad remaining space with zeros
    while (combinedFeatures.size() < EXPECTED_FEATURE_SIZE) {
        combinedFeatures.push_back(0.0);
    }
    
    std::cout << "Combined feature vector size: " << combinedFeatures.size() << "\n";
    
    // Validate final feature vector
    if (combinedFeatures.size() != EXPECTED_FEATURE_SIZE) {
        std::cerr << "Warning: Feature vector size mismatch. Expected: " 
                  << EXPECTED_FEATURE_SIZE << ", Got: " 
                  << combinedFeatures.size() << std::endl;
    }
    
    return combinedFeatures;
}

void GaitAnalyzer::normalizeFeatureVector(std::vector<double>& features, const std::string& name) {
    if (features.empty()) return;
    
    // Find valid min and max values
    double minVal = std::numeric_limits<double>::max();
    double maxVal = -std::numeric_limits<double>::max();
    
    for (const auto& val : features) {
        if (!std::isnan(val) && !std::isinf(val) && std::abs(val) > 1e-20) {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }
    
    std::cout << name << " features before normalization - Range: [" 
              << minVal << ", " << maxVal << "]\n";
    
    // Normalize to [0,1] range
    if (maxVal - minVal > 1e-10) {  // Only normalize if there's meaningful variation
        for (auto& val : features) {
            if (std::isnan(val) || std::isinf(val) || std::abs(val) < 1e-20) {
                val = 0.0;
            } else {
                val = (val - minVal) / (maxVal - minVal);
            }
        }
    } else {
        // If all values are essentially the same, set to a constant value
        std::fill(features.begin(), features.end(), 0.5);
    }
    
    // Print sample of normalized values
    std::cout << name << " features after normalization - First 5 values: ";
    for (size_t i = 0; i < std::min(size_t(5), features.size()); i++) {
        std::cout << features[i] << " ";
    }
    std::cout << "...\n";
}

std::vector<double> GaitAnalyzer::computeFourierDescriptors(const cv::Mat& symmetryMap) {
    std::vector<double> descriptors;
    const int numSamples = 32;  // Reduced number of samples
    
    try {
        // Create complex image for DFT
        cv::Mat complexImg;
        cv::dft(symmetryMap, complexImg, cv::DFT_COMPLEX_OUTPUT);
        
        std::vector<cv::Mat> planes;
        cv::split(complexImg, planes);
        
        // Compute magnitude spectrum
        cv::Mat magnitudeImg;
        cv::magnitude(planes[0], planes[1], magnitudeImg);
        
        // Compute log magnitude with offset to avoid log(0)
        cv::Mat logMagnitude;
        cv::log(magnitudeImg + 1.0, logMagnitude);
        
        // Extract circular samples with more robust sampling
        const double maxRadius = std::min(logMagnitude.rows/2, logMagnitude.cols/2);
        const int angleSteps = 36;  // Sample every 10 degrees
        
        for (int r = 0; r < maxRadius; r += maxRadius/numSamples) {
            double sum = 0.0;
            int count = 0;
            
            for (int theta = 0; theta < angleSteps; theta++) {
                double angle = (2.0 * M_PI * theta) / angleSteps;
                int x = logMagnitude.cols/2 + r * std::cos(angle);
                int y = logMagnitude.rows/2 + r * std::sin(angle);
                
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
            if (maxVal > 1e-6) {
                for (auto& val : descriptors) {
                    val /= maxVal;
                }
            }
        }
        
    } catch (const cv::Exception& e) {
        std::cerr << "Error computing Fourier descriptors: " << e.what() << "\n";
        descriptors.resize(numSamples, 0.0);
    }
    
    return descriptors;
}
} // namespace gait
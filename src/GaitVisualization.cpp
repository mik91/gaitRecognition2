#include "GaitVisualization.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace gait {
namespace visualization {

bool initializeWindows() {
    try {
        // Create windows with specific properties
        cv::namedWindow("Original Frame", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::namedWindow("Symmetry Map", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::namedWindow("Features", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

        // Set initial window sizes
        cv::resizeWindow("Original Frame", 640, 480);
        cv::resizeWindow("Symmetry Map", 640, 480);
        cv::resizeWindow("Features", 800, 400);

        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Failed to initialize visualization windows: " << e.what() << std::endl;
        return false;
    }
}

void cleanupWindows() {
    try {
        cv::destroyAllWindows();
    } catch (const cv::Exception& e) {
        std::cerr << "Error during window cleanup: " << e.what() << std::endl;
    }
}

bool displayResults(const cv::Mat& originalFrame, const cv::Mat& symmetryMap, 
                   const std::vector<double>& features) {
    try {
        // Check if input images are valid
        if (originalFrame.empty() || symmetryMap.empty()) {
            std::cerr << "Invalid input images for visualization" << std::endl;
            return false;
        }

        // Display original frame
        cv::imshow("Original Frame", originalFrame);

        // Visualize and display symmetry map
        cv::Mat symmetryVis = visualizeSymmetryMap(symmetryMap);
        if (!symmetryVis.empty()) {
            cv::imshow("Symmetry Map", symmetryVis);
        }

        // Visualize and display features
        cv::Mat featureVis = visualizeGaitFeatures(features);
        if (!featureVis.empty()) {
            cv::imshow("Features", featureVis);
        }

        // Process UI events and wait for key
        int key = cv::waitKey(30); // 30ms delay for smooth visualization
        return (key != 27); // Return false if ESC pressed

    } catch (const cv::Exception& e) {
        std::cerr << "Visualization error: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat visualizeSymmetryMap(const cv::Mat& symmetryMap) {
    try {
        if (symmetryMap.empty()) {
            std::cerr << "Empty symmetry map provided" << std::endl;
            return cv::Mat();
        }

        cv::Mat normalized, colored;
        
        // Ensure input is float type
        cv::Mat floatMap;
        if (symmetryMap.type() != CV_32F) {
            symmetryMap.convertTo(floatMap, CV_32F);
        } else {
            floatMap = symmetryMap;
        }
        
        // Normalize with error checking
        double minVal, maxVal;
        cv::minMaxLoc(floatMap, &minVal, &maxVal);
        if (std::abs(maxVal - minVal) < 1e-6) {
            floatMap.copyTo(normalized);
        } else {
            cv::normalize(floatMap, normalized, 0, 1, cv::NORM_MINMAX);
        }
        
        // Convert to 8-bit for visualization
        cv::Mat visualMap;
        normalized.convertTo(visualMap, CV_8UC1, 255);
        
        // Apply color map
        cv::applyColorMap(visualMap, colored, cv::COLORMAP_JET);
        
        // Draw contours (optional - comment out if too slow)
        std::vector<std::vector<cv::Point>> contours;
        for (float level = 0.2f; level < 1.0f; level += 0.2f) {
            cv::Mat binary;
            cv::threshold(normalized, binary, level, 1.0, cv::THRESH_BINARY);
            binary.convertTo(binary, CV_8UC1, 255);
            cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(colored, contours, -1, cv::Scalar(255,255,255), 1);
        }
        
        return colored;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error in symmetry map visualization: " << e.what() << std::endl;
        return cv::Mat();
    }
}

cv::Mat visualizeGaitFeatures(const std::vector<double>& features) {
    if (features.empty()) {
        std::cerr << "No features to visualize" << std::endl;
        return cv::Mat();
    }

    // Get current window size - if window doesn't exist, use default size
    cv::Size windowSize;
    try {
        windowSize = cv::getWindowImageRect("Features").size();
    } catch (...) {
        windowSize = cv::Size(800, 400);
    }

    // Create visualization with current window dimensions
    cv::Mat visualization = cv::Mat::zeros(windowSize, CV_8UC3);
    visualization.setTo(cv::Scalar(255, 255, 255)); // White background

    // Draw grid lines efficiently using line arrays
    std::vector<cv::Point> horizontalLines;
    for (int i = 0; i <= 10; i++) {
        int y = static_cast<int>(i * windowSize.height / 10);
        horizontalLines.push_back(cv::Point(0, y));
        horizontalLines.push_back(cv::Point(windowSize.width, y));
    }
    cv::polylines(visualization, horizontalLines, false, cv::Scalar(200, 200, 200), 1);

    // Add legend text once
    cv::putText(visualization, "Regional", cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
    cv::putText(visualization, "Temporal", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    cv::putText(visualization, "Fourier", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

    // Calculate feature statistics once
    double minVal = *std::min_element(features.begin(), features.end());
    double maxVal = *std::max_element(features.begin(), features.end());
    double range = (std::abs(maxVal - minVal) < 1e-10) ? 1.0 : maxVal - minVal;

    // Pre-calculate bar dimensions
    int barWidth = windowSize.width / features.size();
    
    // Create arrays for batch rectangle drawing
    std::vector<cv::Rect> bars;
    std::vector<cv::Scalar> colors;
    std::vector<cv::Rect> borders;

    // Prepare rectangles and colors
    for (size_t i = 0; i < features.size(); i++) {
        double normalizedValue = (features[i] - minVal) / range;
        int barHeight = static_cast<int>(normalizedValue * windowSize.height);
        
        cv::Rect bar(i * barWidth, windowSize.height - barHeight, 
                    barWidth - 1, barHeight);
        bars.push_back(bar);
        
        // Determine color based on feature type
        cv::Scalar color = (i < 4) ? cv::Scalar(255, 0, 0) : 
                          (i == 4) ? cv::Scalar(0, 255, 0) : 
                                    cv::Scalar(0, 0, 255);
        colors.push_back(color);
        borders.push_back(bar);
    }

    // Batch draw filled rectangles
    for (size_t i = 0; i < bars.size(); i++) {
        cv::rectangle(visualization, bars[i], colors[i], cv::FILLED);
        cv::rectangle(visualization, borders[i], cv::Scalar(0, 0, 0), 1);
    }

    return visualization;
}

void plotFeatureDistribution(
    const std::vector<std::vector<double>>& normalFeatures,
    const std::vector<std::vector<double>>& abnormalFeatures) {
    
    if(normalFeatures.empty() || abnormalFeatures.empty() || 
       normalFeatures[0].empty() || abnormalFeatures[0].empty()) {
        std::cerr << "Empty feature sets provided for distribution plotting" << std::endl;
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
        double normalMean = normalSum / static_cast<double>(normalFeatures.size());
        double normalStd = std::sqrt(normalSqSum/static_cast<double>(normalFeatures.size()) - normalMean*normalMean);
        normalStats[i] = {normalMean, normalStd};
        
        // Abnormal stats
        double abnormalSum = 0, abnormalSqSum = 0;
        for(const auto& sample : abnormalFeatures) {
            abnormalSum += sample[i];
            abnormalSqSum += sample[i] * sample[i];
        }
        double abnormalMean = abnormalSum / static_cast<double>(abnormalFeatures.size());
        double abnormalStd = std::sqrt(abnormalSqSum/static_cast<double>(abnormalFeatures.size()) - abnormalMean*abnormalMean);
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
    
    // Ensure valid range
    if (std::abs(globalMax - globalMin) < 1e-10) {
        globalMax = globalMin + 1.0;  // Prevent division by zero
    }
    
    // Draw distribution for each feature
    int featureWidth = static_cast<int>(width / numFeatures);
    for(size_t i = 0; i < numFeatures; i++) {
        int centerX = static_cast<int>((i + 0.5) * featureWidth);
        
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
    
    cv::imshow("Feature Distribution", plot);
    cv::waitKey(1);
}

cv::Mat visualizeRegionalFeatures(const std::vector<double>& regionalFeatures) {
    const int height = 400;
    const int width = 400;
    cv::Mat visualization = cv::Mat::zeros(height, width, CV_8UC3);
    visualization.setTo(cv::Scalar(255, 255, 255));  // White background

    if (regionalFeatures.empty()) {
        cv::putText(visualization, "No regional features available",
                    cv::Point(10, height/2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0));
        return visualization;
    }

    // Draw grid lines
    for (int i = 0; i <= height; i += 50) {
        cv::line(visualization, cv::Point(50, i), cv::Point(width-20, i),
                cv::Scalar(200, 200, 200), 1);
    }

    // Find max value for scaling
    double maxVal = *std::max_element(regionalFeatures.begin(), regionalFeatures.end());
    if (maxVal == 0.0) maxVal = 1.0;  // Prevent division by zero

    const int barStartX = 60;  // Leave space for labels
    const int maxBarWidth = width - barStartX - 70;  // Leave space for values
    const int regionHeight = (height - 40) / regionalFeatures.size();

    // Draw region bars and labels
    for (size_t i = 0; i < regionalFeatures.size(); ++i) {
        const int y = 20 + i * regionHeight;
        const int barWidth = static_cast<int>((regionalFeatures[i] / maxVal) * maxBarWidth);
        
        // Draw bar
        cv::rectangle(visualization, 
                     cv::Point(barStartX, y), 
                     cv::Point(barStartX + barWidth, y + regionHeight/2),
                     cv::Scalar(0, 0, 255),  // Red
                     cv::FILLED);
        
        // Draw bar outline
        cv::rectangle(visualization, 
                     cv::Point(barStartX, y), 
                     cv::Point(barStartX + barWidth, y + regionHeight/2),
                     cv::Scalar(0, 0, 0),  // Black
                     1);

        // Add region label
        std::string regionLabel = "R" + std::to_string(i+1);
        cv::putText(visualization, regionLabel,
                    cv::Point(10, y + regionHeight/3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        // Add value label
        std::string valueLabel = cv::format("%.2f", regionalFeatures[i]);
        cv::putText(visualization, valueLabel,
                    cv::Point(barStartX + barWidth + 5, y + regionHeight/3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    // Add title
    cv::putText(visualization, "Regional Features",
                cv::Point(width/3, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0));

    return visualization;
}

cv::Mat visualizeTemporalFeatures(const std::vector<double>& temporalFeatures) {
    const int height = 300;
    const int width = 600;
    cv::Mat visualization = cv::Mat::zeros(height, width, CV_8UC3);
    visualization.setTo(cv::Scalar(255, 255, 255));

    // Debug information
    std::cout << "Visualizing temporal features. Count: " << temporalFeatures.size() << std::endl;

    if (temporalFeatures.empty()) {
        std::cout << "No temporal features to visualize" << std::endl;
        cv::putText(visualization, "No temporal features available (empty)",
                    cv::Point(10, height/2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0));
        return visualization;
    }

    // Print feature values for debugging
    std::cout << "Temporal feature values: ";
    for (const auto& val : temporalFeatures) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Draw grid
    for (int i = 50; i < height; i += 50) {
        cv::line(visualization, cv::Point(50, i), cv::Point(width-20, i),
                cv::Scalar(200, 200, 200), 1);
    }
    for (int i = 50; i < width; i += 50) {
        cv::line(visualization, cv::Point(i, 50), cv::Point(i, height-20),
                cv::Scalar(200, 200, 200), 1);
    }

    // Draw axes
    cv::line(visualization, cv::Point(50, height-50), cv::Point(width-20, height-50),
             cv::Scalar(0, 0, 0), 2);  // X-axis
    cv::line(visualization, cv::Point(50, 50), cv::Point(50, height-50),
             cv::Scalar(0, 0, 0), 2);  // Y-axis

    // Plot points
    const double maxVal = *std::max_element(temporalFeatures.begin(), temporalFeatures.end());
    const double scale = (height - 100.0) / (maxVal > 0.0 ? maxVal : 1.0);
    
    const int plotWidth = width - 70;
    const double xStep = (temporalFeatures.size() > 1) ? 
                        static_cast<double>(plotWidth - 50) / (temporalFeatures.size() - 1) : 
                        plotWidth / 2.0;

    for (size_t i = 0; i < temporalFeatures.size(); ++i) {
        const int xPos = 50 + static_cast<int>(i * xStep);
        const int yPos = height - 50 - static_cast<int>(temporalFeatures[i] * scale);
        
        // Draw point
        cv::circle(visualization, cv::Point(xPos, yPos), 5, cv::Scalar(0, 0, 255), -1);
        
        // Add value label
        std::string valueStr = cv::format("%.2f", temporalFeatures[i]);
        cv::putText(visualization, valueStr,
                    cv::Point(xPos - 20, yPos - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
    }

    // Add title
    cv::putText(visualization, "Temporal Features",
                cv::Point(width/3, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0));

    return visualization;
}

} // namespace visualization
} // namespace gait
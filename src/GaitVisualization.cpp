#include "GaitVisualization.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace gait {
namespace visualization {

bool initializeWindows() {
    try {
        cv::namedWindow("Original Frame", cv::WINDOW_NORMAL);
        cv::namedWindow("Sobel Edges", cv::WINDOW_NORMAL);
        cv::namedWindow("Symmetry Map", cv::WINDOW_NORMAL);
        cv::namedWindow("Features", cv::WINDOW_NORMAL);

        cv::resizeWindow("Original Frame", 400, 800);
        cv::resizeWindow("Sobel Edges", 400, 800);
        cv::resizeWindow("Symmetry Map", 400, 800);
        cv::resizeWindow("Features", 1500, 300); 

        cv::moveWindow("Original Frame", 0, 0);
        cv::moveWindow("Sobel Edges", 420, 0);
        cv::moveWindow("Symmetry Map", 840, 0);
        cv::moveWindow("Features", 0, 820);

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
        if (originalFrame.empty() || symmetryMap.empty()) {
            std::cerr << "Invalid input images for visualization" << std::endl;
            return false;
        }

        cv::imshow("Original Frame", originalFrame);

        // Compute and display Sobel edges
        cv::Mat gray, sobelX, sobelY, sobelCombined;
        cv::cvtColor(originalFrame, gray, cv::COLOR_BGR2GRAY);
        cv::Sobel(gray, sobelX, CV_16S, 1, 0);
        cv::Sobel(gray, sobelY, CV_16S, 0, 1);
        
        cv::convertScaleAbs(sobelX, sobelX);
        cv::convertScaleAbs(sobelY, sobelY);
        cv::addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobelCombined);
        
        cv::imshow("Sobel Edges", sobelCombined);

        // Visualize and display symmetry map
        cv::Mat symmetryVis = visualizeSymmetryMap(symmetryMap);
        if (!symmetryVis.empty()) {
            cv::imshow("Symmetry Map", symmetryVis);
        }

        // Visualize and display gait features
        cv::Mat featuresVis = visualizeGaitFeatures(features);
        if (!featuresVis.empty()) {
            cv::imshow("Features", featuresVis);
        }

        int key = cv::waitKey(30);
        return (key != 27);

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
        
        cv::Mat floatMap;
        if (symmetryMap.type() != CV_32F) {
            symmetryMap.convertTo(floatMap, CV_32F);
        } else {
            floatMap = symmetryMap;
        }
        
        double minVal, maxVal;
        cv::minMaxLoc(floatMap, &minVal, &maxVal);
        if (std::abs(maxVal - minVal) < 1e-6) {
            floatMap.copyTo(normalized);
        } else {
            cv::normalize(floatMap, normalized, 0, 1, cv::NORM_MINMAX);
        }
        
        cv::Mat visualMap;
        normalized.convertTo(visualMap, CV_8UC1, 255);
        
        colored = cv::Mat::zeros(visualMap.size(), CV_8UC3);
        
        for(int y = 0; y < visualMap.rows; y++) {
            for(int x = 0; x < visualMap.cols; x++) {
                float val = visualMap.at<uchar>(y,x) / 255.0f;
                colored.at<cv::Vec3b>(y,x) = cv::Vec3b(
                    static_cast<uchar>(255 * val),  // B
                    static_cast<uchar>(255 * val),  // G
                    static_cast<uchar>(255 * val)   // R
                );
            }
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
        return cv::Mat();
    }

    const int height = 300;
    const int width = 1500;
    cv::Mat visualization = cv::Mat::zeros(height, width, CV_8UC3);
    visualization.setTo(cv::Scalar(255, 255, 255));

    for (int i = 0; i <= 10; i++) {
        int y = 40 + i * (height - 80) / 10;
        cv::line(visualization, cv::Point(80, y), 
                 cv::Point(width-20, y), 
                 cv::Scalar(240, 240, 240), 1);
    }

    double minVal = *std::min_element(features.begin(), features.end());
    double maxVal = *std::max_element(features.begin(), features.end());
    double range = maxVal - minVal;
    if (range < 1e-10) range = 1.0;

    int numBars = features.size();
    int barWidth = (width - 100) / numBars;
    int gap = 1;

    for (size_t i = 0; i < features.size(); i++) {
        double normalizedValue = (features[i] - minVal) / range;
        int barHeight = static_cast<int>(normalizedValue * (height - 100));
        
        int x = 80 + i * barWidth;
        int y = height - 60 - barHeight;
        
        cv::Scalar color;
        const size_t numRegionalFeatures = features.size() / 3;
        if (i < numRegionalFeatures) {
            color = cv::Scalar(0, 0, 255);  // R
        } else if (i < 2 * numRegionalFeatures) {
            color = cv::Scalar(0, 255, 0);  // G
        } else {
            color = cv::Scalar(255, 0, 0);  // B
        }
        
        cv::rectangle(visualization, 
                     cv::Point(x, y), 
                     cv::Point(x + barWidth - gap, height - 60),
                     color, cv::FILLED);
                     
        cv::rectangle(visualization, 
                     cv::Point(x, y), 
                     cv::Point(x + barWidth - gap, height - 60),
                     cv::Scalar(0, 0, 0), 1);

        if (normalizedValue > 0.1) {
            std::string label = cv::format("%.2f", features[i]);
            cv::putText(visualization, label,
                       cv::Point(x, y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.3,
                       cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }
    }
    int legendY = 25;
    int rectWidth = 20;
    int rectHeight = 20;
    
    cv::rectangle(visualization, 
                 cv::Point(10, legendY-15), 
                 cv::Point(10+rectWidth, legendY+5), 
                 cv::Scalar(0, 0, 255), cv::FILLED);
    cv::putText(visualization, "Regional", cv::Point(40, legendY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    
    cv::rectangle(visualization, 
                 cv::Point(150, legendY-15), 
                 cv::Point(150+rectWidth, legendY+5), 
                 cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(visualization, "Temporal", cv::Point(180, legendY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    cv::rectangle(visualization, 
                 cv::Point(290, legendY-15), 
                 cv::Point(290+rectWidth, legendY+5), 
                 cv::Scalar(255, 0, 0), cv::FILLED);
    cv::putText(visualization, "Fourier", cv::Point(320, legendY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

    cv::line(visualization, cv::Point(80, height-60), 
             cv::Point(width-20, height-60), cv::Scalar(0, 0, 0), 2);
    cv::line(visualization, cv::Point(80, 40), 
             cv::Point(80, height-60), cv::Scalar(0, 0, 0), 2);

    for (int i = 0; i <= 10; i++) {
        double value = minVal + (i * range / 10.0);
        std::string label = cv::format("%.1f", value);
        cv::putText(visualization, label,
                   cv::Point(10, height - 60 - i * (height-100)/10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4,
                   cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    return visualization;
}
} // namespace visualization
} // namespace gait
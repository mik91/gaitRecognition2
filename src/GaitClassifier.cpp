#include "GaitClassifier.h"
#include <iostream>
#include <sstream>
#include <limits>
#include <cmath>

namespace gait {

GaitClassifier::GaitClassifier() : isModelTrained_(false) {}

bool GaitClassifier::analyzePatterns(const std::map<std::string, std::vector<std::vector<double>>>& personFeatures) {
    if (personFeatures.empty()) return false;

    try {
        // Convert to training format
        std::vector<std::vector<double>> allFeatures;
        std::vector<std::string> labels;
        
        // Process each person's sequences
        for (const auto& [personId, sequences] : personFeatures) {
            for (const auto& sequence : sequences) {
                allFeatures.push_back(sequence);
                labels.push_back(personId);
            }
        }

        // Print debug info
        std::cout << "Training data:\n"
                 << "Number of people: " << personFeatures.size() << "\n"
                 << "Total sequences: " << allFeatures.size() << "\n";

        // Store data for classification
        trainingData_ = allFeatures;
        trainingLabels_ = labels;
        isModelTrained_ = true;
        
        // Create visualization window
        visualizeTrainingData();
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in analyzePatterns: " << e.what() << "\n";
        return false;
    }
}

std::pair<std::string, double> GaitClassifier::identifyPerson(const std::vector<double>& testSequence) {
    if (!isModelTrained_ || trainingData_.empty()) {
        return {"unknown", 0.0};
    }

    try {
        // Validate feature vector size
        if (testSequence.empty()) {
            throw std::runtime_error("Empty test sequence");
        }

        // Check if sizes match
        if (testSequence.size() != trainingData_[0].size()) {
            std::cerr << "Feature vector size mismatch. Expected: " 
                     << trainingData_[0].size() << ", Got: " 
                     << testSequence.size() << std::endl;
            return {"unknown", 0.0};
        }

        // Find nearest neighbor
        double minDistance = std::numeric_limits<double>::max();
        std::string bestMatch;

        for (size_t i = 0; i < trainingData_.size(); i++) {
            double distance = computeDistance(testSequence, trainingData_[i]);
            if (distance < minDistance) {
                minDistance = distance;
                bestMatch = trainingLabels_[i];
            }
        }

        // Convert distance to similarity score
        double similarity = 1.0 / (1.0 + minDistance);
        
        // Visualize result
        visualizeClassification(testSequence);
        
        return {bestMatch, similarity};
    }
    catch (const std::exception& e) {
        std::cerr << "Error in identifyPerson: " << e.what() << std::endl;
        return {"unknown", 0.0};
    }
}

void GaitClassifier::visualizeTrainingData(const std::string& windowName) {
    if (!isModelTrained_) return;

    // Create visualization window
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    
    // Create visualization matrix
    int width = 800, height = 400;
    cv::Mat vis = cv::Mat::zeros(height, width, CV_8UC3);
    
    // Get unique labels
    std::set<std::string> uniqueLabels(trainingLabels_.begin(), trainingLabels_.end());
    std::map<std::string, cv::Scalar> colorMap;
    int colorIdx = 0;
    for (const auto& label : uniqueLabels) {
        cv::Scalar color(
            (colorIdx * 90) % 255,
            (colorIdx * 150) % 255,
            (colorIdx * 200) % 255
        );
        colorMap[label] = color;
        colorIdx++;
    }
    
    // Plot features
    for (size_t i = 0; i < trainingData_.size(); i++) {
        const auto& features = trainingData_[i];
        const auto& label = trainingLabels_[i];
        
        // Plot first two dimensions of features
        if (features.size() >= 2) {
            int x = static_cast<int>(features[0] * (width - 20)) + 10;
            int y = static_cast<int>(features[1] * (height - 20)) + 10;
            cv::circle(vis, cv::Point(x, y), 5, colorMap[label], -1);
        }
    }
    
    // Show visualization
    cv::imshow(windowName, vis);
    cv::waitKey(1);
}

void GaitClassifier::visualizeClassification(const std::vector<double>& testSequence, 
                                           const std::string& windowName) {
    // Create visualization matrix
    int width = 800, height = 400;
    cv::Mat vis = cv::Mat::zeros(height, width, CV_8UC3);
    
    // Plot training data points
    for (size_t i = 0; i < trainingData_.size(); i++) {
        const auto& features = trainingData_[i];
        if (features.size() >= 2) {
            int x = static_cast<int>(features[0] * (width - 20)) + 10;
            int y = static_cast<int>(features[1] * (height - 20)) + 10;
            cv::circle(vis, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);
        }
    }
    
    // Plot test sequence point in red
    if (testSequence.size() >= 2) {
        int x = static_cast<int>(testSequence[0] * (width - 20)) + 10;
        int y = static_cast<int>(testSequence[1] * (height - 20)) + 10;
        cv::circle(vis, cv::Point(x, y), 7, cv::Scalar(0, 0, 255), -1);
    }
    
    // Show visualization
    cv::imshow(windowName, vis);
    cv::waitKey(1);
}

double GaitClassifier::computeDistance(const std::vector<double>& seq1, const std::vector<double>& seq2) {
    if (seq1.size() != seq2.size()) {
        throw std::runtime_error("Feature vector size mismatch");
    }

    double sumSquaredDiff = 0.0;
    for (size_t i = 0; i < seq1.size(); i++) {
        double diff = seq1[i] - seq2[i];
        sumSquaredDiff += diff * diff;
    }
    return std::sqrt(sumSquaredDiff);
}

std::string GaitClassifier::getClusterStats() const {
    std::stringstream ss;
    ss << "Cluster Statistics:\n";
    for (size_t i = 0; i < clusterStats_.size(); i++) {
        ss << "Cluster " << (i == 0 ? "A" : "B") << ":\n";
        ss << " Size: " << clusterStats_[i].size << "\n";
        ss << " Average Distance to Center: " << clusterStats_[i].avgDistance << "\n";
        ss << " Max Distance to Center: " << clusterStats_[i].maxDistance << "\n";
        ss << " Variance: " << clusterStats_[i].variance << "\n";
    }
    return ss.str();
}

} // namespace gait
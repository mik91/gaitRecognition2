#include "PersonIdentifier.h"
#include "GaitVisualization.h"
#include "PathConfig.h"
#include <stdexcept>
#include <iostream>
#include <filesystem>

namespace gait {

PersonIdentifier::PersonIdentifier(GaitAnalyzer& analyzer, GaitClassifier& classifier)
    : analyzer_(analyzer), classifier_(classifier) {}

std::pair<std::string, double> PersonIdentifier::identifyFromImage(
    const std::string& imagePath,
    bool visualize) {
    
    // Get default results directory from PathConfig
    auto& config = PathConfig::getInstance();
    std::filesystem::path outputDir(config.getPath("RESULTS_DIR"));
    return identifyFromImage(imagePath, visualize, outputDir.string());
}

std::pair<std::string, double> PersonIdentifier::identifyFromImage(
    const std::string& imagePath,
    bool visualize,
    const std::string& outputDir) {
    
    try {
        // Convert input path to filesystem path
        std::filesystem::path inputPath(imagePath);
        std::filesystem::path resultDir(outputDir);

        // Create output directory if it doesn't exist
        std::filesystem::create_directories(resultDir);

        // Load and validate image
        cv::Mat inputImage = cv::imread(inputPath.string());
        if (inputImage.empty()) {
            throw std::runtime_error("Failed to load image: " + inputPath.string());
        }

        // Process image through GaitAnalyzer
        cv::Mat symmetryMap = analyzer_.processFrame(inputImage);
        std::vector<double> features = analyzer_.extractGaitFeatures(symmetryMap);

        // Debug output for feature vector
        std::cout << "Feature vector composition:\n"
                  << "Total features: " << features.size() << "\n"
                  << "First 4 (Regional): ";
        for (int i = 0; i < 4 && i < features.size(); i++) {
            std::cout << features[i] << " ";
        }
        std::cout << "\nNext 3 (Temporal): ";
        for (int i = 4; i < 7 && i < features.size(); i++) {
            std::cout << features[i] << " ";
        }
        std::cout << "\nRemaining (Fourier): " 
                  << (features.size() > 7 ? features.size() - 7 : 0) << " features\n";

        // Validate feature vector
        if (features.empty()) {
            throw std::runtime_error("No features extracted from the image");
        }

        // Validate feature values
        for (size_t i = 0; i < features.size(); ++i) {
            if (std::isnan(features[i]) || std::isinf(features[i])) {
                std::cout << "Warning: Invalid feature value at index " << i 
                         << ", replacing with 0" << std::endl;
                features[i] = 0.0;
            }
        }

        // Use classifier to identify person
        auto [personId, confidence] = classifier_.identifyPerson(features);

        // Save results
        saveResults(inputImage, symmetryMap, personId, confidence, resultDir.string());

        // Handle visualization if requested
        if (visualize) {
            cv::namedWindow("Input Image", cv::WINDOW_NORMAL);
            cv::namedWindow("Symmetry Map", cv::WINDOW_NORMAL);
            cv::namedWindow("Result", cv::WINDOW_NORMAL);
            
            cv::Mat resultDisplay = inputImage.clone();
            std::string resultText = "Predicted: " + personId;
            std::string confidenceText = "Confidence: " + std::to_string(confidence);
            cv::putText(resultDisplay, resultText, cv::Point(20, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(resultDisplay, confidenceText, cv::Point(20, 70), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Input Image", inputImage);
            cv::imshow("Symmetry Map", visualization::visualizeSymmetryMap(symmetryMap));
            cv::imshow("Result", resultDisplay);
            
            cv::waitKey(1);
        }
        
        return {personId, confidence};
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing image: " << e.what() << std::endl;
        return {"unknown", 0.0};
    }
}

void PersonIdentifier::saveResults(
    const cv::Mat& inputImage,
    const cv::Mat& symmetryMap,
    const std::string& personId,
    double confidence,
    const std::string& outputDir) {
    
    std::filesystem::path resultDir(outputDir);
    
    // Create result visualization
    cv::Mat resultDisplay = inputImage.clone();
    std::string resultText = "Predicted: " + personId;
    std::string confidenceText = "Confidence: " + std::to_string(confidence);
    cv::putText(resultDisplay, resultText, cv::Point(20, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::putText(resultDisplay, confidenceText, cv::Point(20, 70), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    // Save all results
    try {
        cv::imwrite((resultDir / "result.png").string(), resultDisplay);
        cv::imwrite((resultDir / "symmetry_map.png").string(), 
                   visualization::visualizeSymmetryMap(symmetryMap));
        
        // Get current time
        std::time_t currentTime = std::time(nullptr);
        
        // Save metadata
        std::ofstream metaFile(resultDir / "metadata.txt");
        metaFile << "Prediction Results\n"
                 << "=================\n"
                 << "Predicted Person: " << personId << "\n"
                 << "Confidence: " << confidence << "\n"
                 << "Generated: " << std::put_time(std::localtime(&currentTime), 
                                                 "%Y-%m-%d %H:%M:%S") << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << std::endl;
    }
}
} // namespace gait
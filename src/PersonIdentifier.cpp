#include "PersonIdentifier.h"
#include "GaitVisualization.h"
#include <stdexcept>
#include <iostream>
#include <PathConfig.h>
#include <filesystem>

namespace gait {

PersonIdentifier::PersonIdentifier(GaitAnalyzer& analyzer, GaitClassifier& classifier)
    : analyzer_(analyzer), classifier_(classifier) {}

std::pair<std::string, double> PersonIdentifier::identifyFromImage(const std::string& imagePath, bool visualize) {
    try {
        // Convert input path to filesystem path for cross-platform handling
        std::filesystem::path inputPath(imagePath);
        auto& config = PathConfig::getInstance();
        std::filesystem::path outputDir(config.getPath("RESULTS_DIR"));

        // Create output directory if it doesn't exist
        std::filesystem::create_directories(outputDir);

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
        std::cout << "\nRemaining (Fourier): " << (features.size() > 7 ? features.size() - 7 : 0) << " features\n";

        // Validate feature vector
        if (features.empty()) {
            throw std::runtime_error("No features extracted from the image");
        }

        // Validate feature values
        for (size_t i = 0; i < features.size(); ++i) {
            if (std::isnan(features[i]) || std::isinf(features[i])) {
                std::cout << "Warning: Invalid feature value at index " << i << ", replacing with 0" << std::endl;
                features[i] = 0.0;
            }
        }

        // Use classifier to identify person
        try {
            auto [personId, confidence] = classifier_.identifyPerson(features);
            
            // Create result visualization
            cv::Mat resultDisplay = inputImage.clone();
            std::string resultText = "Predicted: " + personId;
            std::string confidenceText = "Confidence: " + std::to_string(confidence);
            cv::putText(resultDisplay, resultText, cv::Point(20, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(resultDisplay, confidenceText, cv::Point(20, 70), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

            // Handle visualization if requested
            if (visualize) {
                cv::namedWindow("Input Image", cv::WINDOW_NORMAL);
                cv::namedWindow("Symmetry Map", cv::WINDOW_NORMAL);
                cv::namedWindow("Result", cv::WINDOW_NORMAL);
                
                cv::imshow("Input Image", inputImage);
                cv::imshow("Symmetry Map", visualization::visualizeSymmetryMap(symmetryMap));
                cv::imshow("Result", resultDisplay);
                
                cv::waitKey(1);
            }
            
            // Save result image with proper path handling
            std::filesystem::path outputPath = outputDir / 
                (inputPath.stem().string() + "_result" + inputPath.extension().string());

            if (cv::imwrite(outputPath.string(), resultDisplay)) {
                std::cout << "Result saved as: " << outputPath << std::endl;
            } else {
                std::cerr << "Warning: Failed to save result image to: " << outputPath << std::endl;
            }
            
            return {personId, confidence};
        }
        catch (const std::runtime_error& e) {
            std::cerr << "Error in identification: " << e.what() << std::endl;
            return {"unknown", 0.0};
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing image: " << e.what() << std::endl;
        return {"unknown", 0.0};
    }
}

} // namespace gait
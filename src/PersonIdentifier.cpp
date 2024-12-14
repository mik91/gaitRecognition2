// PersonIdentifier.cpp
#include "PersonIdentifier.h"
#include "GaitVisualization.h"
#include "PathConfig.h"
#include "FeatureHandler.h"
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
        // Load and process image
        cv::Mat inputImage = cv::imread(imagePath);
        if (inputImage.empty()) {
            throw std::runtime_error("Failed to load image: " + imagePath);
        }

        std::vector<cv::Mat> symmetryMaps;
        std::vector<double> accumulatedFeatures = processImages({inputImage}, visualize, &symmetryMaps);
        
        if (accumulatedFeatures.empty()) {
            throw std::runtime_error("Failed to extract features from image");
        }

        // Use classifier with accumulated features
        auto [personId, confidence] = classifier_.identifyPerson(accumulatedFeatures, 
                                                               std::filesystem::path(imagePath).filename().string());

        // Save results if visualization is requested
        if (visualize && !symmetryMaps.empty()) {
            saveResults(inputImage, symmetryMaps[0], personId, confidence, outputDir);
        }
        
        return {personId, confidence};
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing image: " << e.what() << std::endl;
        return {"unknown", 0.0};
    }
}

std::pair<std::string, double> PersonIdentifier::identifyFromSequence(
    const std::vector<std::string>& imagePaths,
    bool visualize) {
    
    if (imagePaths.empty()) {
        return {"unknown", 0.0};
    }

    try {
        std::vector<cv::Mat> images;
        images.reserve(imagePaths.size());

        // Load all images
        for (const auto& path : imagePaths) {
            cv::Mat img = cv::imread(path);
            if (!img.empty()) {
                images.push_back(img);
            }
        }

        if (images.empty()) {
            throw std::runtime_error("No valid images loaded from sequence");
        }

        // Process all images and accumulate features
        std::vector<double> accumulatedFeatures = processImages(images, visualize);
        
        if (accumulatedFeatures.empty()) {
            throw std::runtime_error("Failed to extract features from sequence");
        }

        // Use first image path as reference for condition information
        return classifier_.identifyPerson(accumulatedFeatures, 
                                        std::filesystem::path(imagePaths[0]).filename().string());
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing sequence: " << e.what() << std::endl;
        return {"unknown", 0.0};
    }
}

std::vector<double> PersonIdentifier::processImages(
    const std::vector<cv::Mat>& images,
    bool visualize,
    std::vector<cv::Mat>* symmetryMaps) {
    
    std::vector<std::vector<double>> allFeatures;
    
    for (const auto& image : images) {
        cv::Mat symmetryMap = analyzer_.processFrame(image);
        if (symmetryMap.empty()) {
            continue;
        }

        if (symmetryMaps) {
            symmetryMaps->push_back(symmetryMap);
        }

        std::vector<double> features = analyzer_.extractGaitFeatures(symmetryMap);
        if (!features.empty()) {
            allFeatures.push_back(features);
        }

        if (visualize) {
            gait::visualization::displayResults(image, symmetryMap, features);
        }
    }

    // Use FeatureHandler to normalize and accumulate features
    return FeatureHandler::normalizeAndResampleFeatures(allFeatures);
}

void PersonIdentifier::saveResults(
    const cv::Mat& inputImage,
    const cv::Mat& symmetryMap,
    const std::string& personId,
    double confidence,
    const std::string& outputDir) {
    
    std::filesystem::path resultDir(outputDir);
    
    try {
        std::filesystem::create_directories(resultDir);
        
        // Create result visualization
        cv::Mat resultDisplay = inputImage.clone();
        std::string resultText = "Predicted: " + personId;
        std::string confidenceText = "Confidence: " + std::to_string(confidence);
        
        cv::putText(resultDisplay, resultText, cv::Point(20, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(resultDisplay, confidenceText, cv::Point(20, 70), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // Save results
        cv::imwrite((resultDir / "result.png").string(), resultDisplay);
        cv::imwrite((resultDir / "symmetry_map.png").string(), 
                   visualization::visualizeSymmetryMap(symmetryMap));
        
        // Save metadata
        std::time_t currentTime = std::time(nullptr);
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
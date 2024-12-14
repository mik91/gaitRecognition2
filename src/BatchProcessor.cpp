#include "BatchProcessor.h"
#include "PathConfig.h"
#include "FeatureHandler.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <ctime>

namespace gait {

// In BatchProcessor.cpp
std::vector<BatchProcessor::ProcessingResult> BatchProcessor::processDirectory(
    const std::string& inputDir,
    bool visualize,
    const std::vector<std::string>& validExtensions) {
    
    std::vector<ProcessingResult> results;
    std::vector<cv::Mat> allImages;
    std::vector<std::string> filenames;

    // Add error checking
    if (!std::filesystem::exists(inputDir)) {
        std::cerr << "Directory does not exist: " << inputDir << std::endl;
        return results;
    }

    // First collect all valid images with error checking
    try {
        for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                if (std::find(validExtensions.begin(), validExtensions.end(), extension) 
                    != validExtensions.end()) {
                    
                    cv::Mat img = cv::imread(entry.path().string());
                    if (!img.empty()) {
                        allImages.push_back(img);
                        filenames.push_back(entry.path().filename().string());
                        // std::cout << "Loaded image: " << entry.path().filename().string() << std::endl;
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error processing directory: " << e.what() << std::endl;
        return results;
    }

    if (allImages.empty()) {
        std::cerr << "No valid images found in directory" << std::endl;
        return results;
    }

    // Process all images and accumulate features with bounds checking
    std::vector<std::vector<double>> allFeatures;
    for (size_t i = 0; i < allImages.size(); i++) {
        try {
            if (i == 95) {
                std::cout << "Processing images...\n";
            }   
            cv::Mat symmetryMap = analyzer_.processFrame(allImages[i]);
            // std::cout << "Processed image " << filenames[i] << std::endl;
            if (!symmetryMap.empty()) {
                std::vector<double> frameFeatures = analyzer_.extractGaitFeatures(symmetryMap);
                if (!frameFeatures.empty()) {
                    allFeatures.push_back(frameFeatures);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing image " << filenames[i] << ": " << e.what() << std::endl;
            continue;
        }
    }

    if (allFeatures.empty()) {
        std::cerr << "No features could be extracted from images" << std::endl;
        return results;
    }

    try {
        // Accumulate features for the whole sequence
        std::vector<double> accumulatedFeatures = FeatureHandler::normalizeAndResampleFeatures(allFeatures);  

        if (!accumulatedFeatures.empty()) {
            // Get first filename as representative for sequence
            std::string representative_filename = filenames[0];
            
            // Get single prediction for all images using the representative filename
            auto [predictedPerson, confidence] = classifier_.identifyPerson(
                accumulatedFeatures, 
                representative_filename  // Pass the representative filename
            );
            
            // Create individual results but use the same prediction
            for (const auto& filename : filenames) {
                ProcessingResult result;
                result.filename = filename;
                result.predictedPerson = predictedPerson;
                result.confidence = confidence;
                result.processingTime = 0; // Set actual processing time if needed
                results.push_back(result);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during classification: " << e.what() << std::endl;
        return results;
    }

    // Save results and write summary
    if (!results.empty()) {
        std::filesystem::path outputPath = std::filesystem::path(inputDir) / "batch_results.csv";
        std::ofstream outputFile(outputPath);
        if (outputFile.is_open()) {
            writeSummaryReport(results, outputFile);
        }
        summarizeResults(results);
    }

    return results;
}

void BatchProcessor::summarizeResults(const std::vector<ProcessingResult>& results) {
    if (results.empty()) {
        std::cout << "No images were processed.\n";
        return;
    }

    // Calculate statistics
    double totalConfidence = 0.0;
    double totalTime = 0.0;
    std::map<std::string, int> personCounts;
    
    for (const auto& result : results) {
        totalConfidence += result.confidence;
        totalTime += result.processingTime;
        personCounts[result.predictedPerson]++;
    }

    // Define threshold for minimum percentage to be considered valid
    const double MIN_PERCENTAGE_THRESHOLD = 30.0; // 30%

    // Find the most frequent prediction
    std::string mostLikelyMatch = "unknown";
    double highestPercentage = 0.0;
    double avgConfidence = 0.0;

    for (const auto& [person, count] : personCounts) {
        double percentage = (100.0 * count) / results.size();
        if (percentage > highestPercentage) {
            highestPercentage = percentage;
            // Only assign most likely match if it meets the threshold
            if (percentage >= MIN_PERCENTAGE_THRESHOLD) {
                mostLikelyMatch = person;
                // Calculate average confidence for this person
                double personConfidence = 0.0;
                int confCount = 0;
                for (const auto& result : results) {
                    if (result.predictedPerson == person) {
                        personConfidence += result.confidence;
                        confCount++;
                    }
                }
                avgConfidence = personConfidence / confCount;
            }
        }
    }

    // Print summary with adjusted output
    std::cout << "\nTotal images processed: " << results.size() << "\n"
              << "Most likely match: " << mostLikelyMatch 
              << " (" << std::fixed << std::setprecision(1) 
              << (mostLikelyMatch != "unknown" ? highestPercentage : 0.0) 
              << "% of predictions, "
              << std::setprecision(4) << avgConfidence << " avg confidence)\n"
              << "Average confidence: " << (totalConfidence / results.size()) << "\n"
              << "Average processing time: " << (totalTime / results.size()) << " ms\n";
}

void BatchProcessor::writeSummaryReport(
    const std::vector<ProcessingResult>& results, 
    std::ofstream& file) {
    
    if (!file.is_open()) {
        std::cerr << "Could not write summary report - file not open" << std::endl;
        return;
    }

    // Get current time
    std::time_t currentTime = std::time(nullptr);
    
    // Write header
    file << "Batch Processing Summary Report\n"
         << "==============================\n"
         << "Generated: " << std::put_time(std::localtime(&currentTime), 
                                         "%Y-%m-%d %H:%M:%S") << "\n\n";

    if (results.empty()) {
        file << "No images were processed.\n";
        return;
    }

    // Calculate statistics
    double totalConfidence = 0.0;
    double totalTime = 0.0;
    std::map<std::string, int> personCounts;
    double minConfidence = 1.0;
    double maxConfidence = 0.0;
    double minTime = std::numeric_limits<double>::max();
    double maxTime = 0.0;

    for (const auto& result : results) {
        totalConfidence += result.confidence;
        totalTime += result.processingTime;
        personCounts[result.predictedPerson]++;
        
        minConfidence = std::min(minConfidence, result.confidence);
        maxConfidence = std::max(maxConfidence, result.confidence);
        minTime = std::min(minTime, result.processingTime);
        maxTime = std::max(maxTime, result.processingTime);
    }

    // Write statistics
    file << "Overall Statistics:\n"
         << "-----------------\n"
         << "Total images processed: " << results.size() << "\n"
         << "Average confidence: " << (totalConfidence / results.size()) << "\n"
         << "Confidence range: " << minConfidence << " - " << maxConfidence << "\n"
         << "Average processing time: " << (totalTime / results.size()) << " ms\n"
         << "Processing time range: " << minTime << " - " << maxTime << " ms\n\n";

    // Write predictions breakdown
    file << "Predictions Breakdown:\n"
         << "--------------------\n";
    for (const auto& [person, count] : personCounts) {
        double percentage = (100.0 * count) / results.size();
        file << person << ": " << count << " images (" 
             << std::fixed << std::setprecision(1) << percentage << "%)\n";
    }
    file << "\n";

    // Write detailed results
    file << "Detailed Results:\n"
         << "---------------\n";
    for (const auto& result : results) {
        file << "File: " << result.filename << "\n"
             << "  Predicted Person: " << result.predictedPerson << "\n"
             << "  Confidence: " << std::fixed << std::setprecision(4) 
             << result.confidence << "\n"
             << "  Processing Time: " << result.processingTime << " ms\n\n";
    }
}

} // namespace gait
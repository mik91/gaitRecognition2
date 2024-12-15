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
    std::vector<cv::Mat> frames;
    std::vector<std::string> filenames;

    // Loading files phase
    std::cout << "\nCollecting files..." << std::endl;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                if (std::find(validExtensions.begin(), validExtensions.end(), extension) 
                    != validExtensions.end()) {
                    
                    cv::Mat img = cv::imread(entry.path().string());
                    if (!img.empty()) {
                        frames.push_back(img);
                        filenames.push_back(entry.path().filename().string());
                    }
                }
            }
        }
        std::cout << "Found " << frames.size() << " valid images" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error collecting files: " << e.what() << std::endl;
        return results;
    }

    // Processing phase
    std::vector<std::vector<double>> allFeatures;
    allFeatures.reserve(frames.size());

    std::cout << "\nProcessing frames..." << std::endl;
    for (size_t i = 0; i < frames.size(); i++) {
        if (i % 5 == 0) {  // Update more frequently
            std::cout << "\rFrame " << i+1 << "/" << frames.size() 
                     << " (" << (i * 100) / frames.size() << "%)" << std::flush;
        }
        
        cv::Mat symmetryMap = analyzer_.processFrame(frames[i]);
        if (!symmetryMap.empty()) {
            std::vector<double> frameFeatures = analyzer_.extractGaitFeatures(symmetryMap);
            if (!frameFeatures.empty()) {
                allFeatures.push_back(frameFeatures);
            }
        }
    }
    std::cout << "\nProcessed " << allFeatures.size() << " frames successfully" << std::endl;

    // Classification phase
    if (!allFeatures.empty()) {
        std::cout << "\nGenerating sequence features..." << std::endl;
        std::vector<double> accumulatedFeatures = 
            FeatureHandler::normalizeAndResampleFeatures(allFeatures);
            
        std::cout << "Classifying sequence..." << std::endl;
        auto classifyStart = std::chrono::steady_clock::now();
        
        auto [predictedPerson, confidence] = classifier_.identifyPerson(
            accumulatedFeatures, filenames[0]);
            
        auto classifyDuration = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - classifyStart).count();
            
        std::cout << "\nClassification Results:" << std::endl;
        std::cout << "------------------------" << std::endl;
        if (confidence == 0.0) {
            std::cout << "Result: Unknown/Rejected" << std::endl;
            std::cout << "Reason: ";
            if (predictedPerson == "unknown") {
                std::cout << "Subject not in database" << std::endl;
            } else {
                std::cout << "Low confidence match with " << predictedPerson 
                        << " (possible different condition)" << std::endl;
            }
        } else {
            std::cout << "Predicted Person: " << predictedPerson << std::endl;
            std::cout << "Confidence: " << std::fixed << std::setprecision(4) 
                    << confidence << std::endl;
            std::cout << "Status: " << (confidence > 0.85 ? "Strong match" : "Moderate match") 
                    << std::endl;
        }
        std::cout << "Classification time: " << classifyDuration << " seconds" << std::endl;
        
        // Create results
        for (const auto& filename : filenames) {
            ProcessingResult result;
            result.filename = filename;
            result.predictedPerson = predictedPerson;
            result.confidence = confidence;
            result.processingTime = classifyDuration;
            results.push_back(result);
        }

        // Immediately write results
        auto& config = PathConfig::getInstance();
        std::filesystem::path outputPath = config.getPath("RESULTS_DIR") + "/batch_results.txt";
        std::ofstream outputFile(outputPath);
        if (outputFile.is_open()) {
            writeSummaryReport(results, outputFile);
            std::cout << "\nResults saved to: " << outputPath << std::endl;
        } else {
            std::cerr << "Failed to save results to file" << std::endl;
        }
        
        // Print summary
        std::cout << "\nSummary:" << std::endl;
        std::cout << "--------" << std::endl;
        std::cout << "Total images processed: " << results.size() << std::endl;
        std::cout << "Most likely match: " << predictedPerson << std::endl;
        std::cout << "Confidence: " << confidence << std::endl;
    } else {
        std::cerr << "No features could be extracted from images" << std::endl;
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
    std::map<std::string, int> personCounts;
    
    for (const auto& result : results) {
        totalConfidence += result.confidence;
        personCounts[result.predictedPerson]++;
    }

    // Calculate percentages and gather stats
    std::vector<std::pair<std::string, std::pair<int, double>>> predictions;
    for (const auto& [person, count] : personCounts) {
        double percentage = (100.0 * count) / results.size();
        double avgConfidence = 0.0;
        int confCount = 0;
        
        for (const auto& result : results) {
            if (result.predictedPerson == person) {
                avgConfidence += result.confidence;
                confCount++;
            }
        }
        avgConfidence /= confCount;
        
        predictions.push_back({person, {count, avgConfidence}});
    }

    // Sort predictions by count
    std::sort(predictions.begin(), predictions.end(),
              [](const auto& a, const auto& b) { 
                  return a.second.first > b.second.first; 
              });

    // Print detailed summary
    std::cout << "\nDetailed Analysis Results:" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "Total frames processed: " << results.size() << std::endl;
    std::cout << "\nPredictions breakdown:" << std::endl;
    
    for (const auto& [person, stats] : predictions) {
        double percentage = (100.0 * stats.first) / results.size();
        std::cout << std::fixed << std::setprecision(2);
        
        if (person == "unknown") {
            std::cout << "Rejected frames: " << stats.first 
                     << " (" << percentage << "%) - likely unknown or different condition" << std::endl;
        } else if (stats.second > 0.0) {
            std::cout << "Person " << person << ": " << stats.first 
                     << " frames (" << percentage << "%) with avg confidence: " 
                     << std::setprecision(4) << stats.second << std::endl;
        }
    }

    // Print final conclusion
    std::cout << "\nFinal conclusion: ";
    if (!predictions.empty() && predictions[0].second.second > 0.0) {
        double mainPredictionPercentage = (100.0 * predictions[0].second.first) / results.size();
        if (mainPredictionPercentage > 70.0) {
            std::cout << "Strong match with " << predictions[0].first 
                     << " (confidence: " << std::setprecision(4) 
                     << predictions[0].second.second << ")" << std::endl;
        } else {
            std::cout << "Uncertain match - possible different condition or unknown subject" 
                     << std::endl;
        }
    } else {
        std::cout << "Unknown subject or different condition" << std::endl;
    }
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
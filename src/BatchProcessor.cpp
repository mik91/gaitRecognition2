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
#include <regex>

namespace gait {

std::vector<BatchProcessor::ProcessingResult> BatchProcessor::processDirectory(
    const std::string& inputDir,
    bool visualize,
    const std::vector<std::string>& validExtensions) {
    
    std::vector<ProcessingResult> results;
    std::vector<cv::Mat> frames;
    std::vector<std::string> filenames;

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
        
        for (const auto& filename : filenames) {
            ProcessingResult result;
            result.filename = filename;
            result.predictedPerson = predictedPerson;
            result.confidence = confidence;
            result.processingTime = classifyDuration;
            results.push_back(result);
        }

        auto& config = PathConfig::getInstance();
        std::filesystem::path outputPath = config.getPath("RESULTS_DIR") + "/batch_results.txt";
        std::ofstream outputFile(outputPath);
        if (outputFile.is_open()) {
            writeSummaryReport(results, outputFile);
            std::cout << "\nResults saved to: " << outputPath << std::endl;
        } else {
            std::cerr << "Failed to save results to file" << std::endl;
        }
        
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

    double totalConfidence = 0.0;
    std::map<std::string, int> personCounts;
    
    for (const auto& result : results) {
        totalConfidence += result.confidence;
        personCounts[result.predictedPerson]++;
    }

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

    std::sort(predictions.begin(), predictions.end(),
              [](const auto& a, const auto& b) { 
                  return a.second.first > b.second.first; 
              });

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
    
    if (!file.is_open() || results.empty()) {
        std::cerr << "Cannot write report - invalid file or empty results" << std::endl;
        return;
    }

    try {
        struct Stats {
            double totalConfidence = 0.0;
            double totalTime = 0.0;
            double minConfidence = 1.0;
            double maxConfidence = 0.0;
            double minTime = std::numeric_limits<double>::max();
            double maxTime = 0.0;
            int highConfCount = 0;    // >0.85
            int modConfCount = 0;     // 0.70-0.85
            int lowConfCount = 0;     // <0.70
            std::map<std::string, int> predictionCounts;
        } stats;

        for (const auto& result : results) {
            stats.totalConfidence += result.confidence;
            stats.totalTime += result.processingTime;
            stats.minConfidence = std::min(stats.minConfidence, result.confidence);
            stats.maxConfidence = std::max(stats.maxConfidence, result.confidence);
            stats.minTime = std::min(stats.minTime, result.processingTime);
            stats.maxTime = std::max(stats.maxTime, result.processingTime);
            stats.predictionCounts[result.predictedPerson]++;

            if (result.confidence > 0.85) stats.highConfCount++;
            else if (result.confidence > 0.70) stats.modConfCount++;
            else stats.lowConfCount++;
        }

        auto primaryPrediction = std::max_element(
            stats.predictionCounts.begin(), 
            stats.predictionCounts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );

        double avgConfidence = stats.totalConfidence / results.size();
        double avgTime = stats.totalTime / results.size();

        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        file << "Batch Processing Summary Report\n"
             << "==============================\n"
             << "Generated: " << std::put_time(std::localtime(&now_time), 
                                             "%Y-%m-%d %H:%M:%S") << "\n\n";

        file << "Overall Statistics\n"
             << "-----------------\n"
             << "Total images processed: " << results.size() << "\n"
             << "Sequence length: " << results.size() << " frames\n"
             << "Processing duration: " << stats.totalTime << " seconds\n\n";

        file << "Classification Results\n"
             << "--------------------\n"
             << "Primary prediction: " << primaryPrediction->first << "\n"
             << "Confidence level: " << std::fixed << std::setprecision(4) << avgConfidence << "\n"
             << "Status: " << (avgConfidence > 0.85 ? "Strong match" : 
                             avgConfidence > 0.70 ? "Moderate match" : "Weak match") << "\n\n";

        file << "Prediction Details\n"
             << "----------------\n"
             << "Confidence distribution:\n"
             << "- High confidence matches (>0.85): " << stats.highConfCount << "\n"
             << "- Moderate confidence matches (0.70-0.85): " << stats.modConfCount << "\n"
             << "- Low confidence matches (<0.70): " << stats.lowConfCount << "\n\n";

        file << "Performance Metrics\n"
             << "-----------------\n"
             << "Average processing time per frame: " << std::fixed << std::setprecision(2) 
             << avgTime << " ms\n"
             << "Processing time range: " << stats.minTime << " - " << stats.maxTime << " ms\n\n";

        if (!results.empty()) {
            file << "Sequence Analysis\n"
                 << "---------------\n"
                 << "Start frame: " << results.front().filename << "\n"
                 << "End frame: " << results.back().filename << "\n";
            
            std::string condition = extractConditionFromFilename(results.front().filename);
            file << "Sequence condition: " << condition << "\n\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error generating report: " << e.what() << std::endl;
        file << "\nError occurred while generating full report.\n";
    }
}

std::string BatchProcessor::extractConditionFromFilename(const std::string& filename) {
    std::regex pattern("\\d{3}-(\\w+)-\\d+");
    std::smatch matches;
    if (std::regex_search(filename, matches, pattern) && matches.size() > 1) {
        return matches[1].str();
    }
    return "unknown";
}
} // namespace gait
#include "BatchProcessor.h"
#include "PathConfig.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <ctime>

namespace gait {

BatchProcessor::BatchProcessor(GaitAnalyzer& analyzer, GaitClassifier& classifier)
    : identifier_(analyzer, classifier) {}

std::vector<BatchProcessor::ProcessingResult> BatchProcessor::processDirectory(
    const std::string& inputDir,
    bool visualize,
    const std::vector<std::string>& validExtensions) {
    
    std::vector<ProcessingResult> results;
    std::filesystem::path inputPath(inputDir);

    if (!std::filesystem::exists(inputPath)) {
        throw std::runtime_error("Input directory does not exist: " + inputDir);
    }

    // Get results directory from PathConfig
    auto& config = PathConfig::getInstance();
    std::filesystem::path resultsDir(config.getPath("RESULTS_DIR"));

    // Create timestamp for this batch
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    
    // Create batch-specific directory
    std::filesystem::path batchDir = resultsDir / ("batch_" + timestamp.str());
    std::filesystem::create_directories(batchDir);

    // Create CSV file for results
    std::filesystem::path resultsFile = batchDir / "batch_results.csv";
    std::ofstream csv(resultsFile);
    csv << "Filename,Predicted Person,Confidence,Processing Time (ms)\n";

    // Count total valid files first for progress reporting
    size_t totalFiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(inputPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (std::find(validExtensions.begin(), validExtensions.end(), extension) 
                != validExtensions.end()) {
                totalFiles++;
            }
        }
    }

    // Process all images in directory
    size_t processedFiles = 0;
    for (const auto& entry : std::filesystem::directory_iterator(inputPath)) {
        if (!entry.is_regular_file()) continue;
        
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

        // Check if file has valid image extension
        if (std::find(validExtensions.begin(), validExtensions.end(), extension) 
            != validExtensions.end()) {
            
            ProcessingResult result;
            result.filename = entry.path().filename().string();

            // Measure processing time
            auto start = std::chrono::high_resolution_clock::now();

            try {
                // Create result-specific directory name using original filename
                std::filesystem::path imageResultDir = batchDir / 
                    (entry.path().stem().string() + "_result");
                std::filesystem::create_directories(imageResultDir);

                // Process the image and save results in the specific directory
                auto [person, conf] = identifier_.identifyFromImage(
                    entry.path().string(), 
                    visualize,
                    imageResultDir.string()  // Pass the specific result directory
                );
                
                result.predictedPerson = person;
                result.confidence = conf;

                auto end = std::chrono::high_resolution_clock::now();
                result.processingTime = 
                    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                // Write to CSV
                csv << result.filename << ","
                    << result.predictedPerson << ","
                    << std::fixed << std::setprecision(4) << result.confidence << ","
                    << result.processingTime << "\n";

                results.push_back(result);

                // Update progress
                processedFiles++;
                float progress = (processedFiles * 100.0f) / totalFiles;
                
                // Print progress bar
                std::cout << "\rProgress: [";
                int barWidth = 50;
                int pos = barWidth * progress / 100.0f;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(progress) << "% "
                         << "(" << processedFiles << "/" << totalFiles << ") "
                         << result.filename << " -> " << result.predictedPerson
                         << std::flush;

            } catch (const std::exception& e) {
                std::cerr << "\nError processing " << entry.path().filename() 
                         << ": " << e.what() << "\n";
            }
        }
    }
    std::cout << std::endl;  // New line after progress bar

    // Save summary report
    std::ofstream summaryFile(batchDir / "summary.txt");
    writeSummaryReport(results, summaryFile);

    // Print summary to console
    summarizeResults(results);

    std::cout << "\nResults saved in: " << batchDir << std::endl;
    
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
    std::map<std::string, std::pair<int, double>> personStats;  // Count and total confidence
    double minConfidence = 1.0;
    double maxConfidence = 0.0;
    double minTime = std::numeric_limits<double>::max();
    double maxTime = 0.0;

    // Collect stats for each prediction
    for (const auto& result : results) {
        totalConfidence += result.confidence;
        totalTime += result.processingTime;
        
        auto& stats = personStats[result.predictedPerson];
        stats.first++;  // Increment count
        stats.second += result.confidence;  // Add confidence
        
        minConfidence = std::min(minConfidence, result.confidence);
        maxConfidence = std::max(maxConfidence, result.confidence);
        minTime = std::min(minTime, result.processingTime);
        maxTime = std::max(maxTime, result.processingTime);
    }

    // Find most likely match
    std::string bestMatch;
    double bestAverageConfidence = 0.0;
    int bestCount = 0;

    for (const auto& [person, stats] : personStats) {
        double avgConfidence = stats.second / stats.first;
        if (avgConfidence > bestAverageConfidence) {
            bestAverageConfidence = avgConfidence;
            bestMatch = person;
            bestCount = stats.first;
        }
    }

    // Print summary
    std::cout << "\nBatch Processing Summary\n"
              << "========================\n"
              << "Total images processed: " << results.size() << "\n"
              << "Most likely match: " << bestMatch << " (" 
              << std::fixed << std::setprecision(1) 
              << (bestCount * 100.0 / results.size()) << "% of predictions, "
              << std::setprecision(4) << bestAverageConfidence << " avg confidence)\n"
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
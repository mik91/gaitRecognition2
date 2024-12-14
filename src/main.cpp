#include "Loader.h"
#include "GaitAnalyzer.h"
#include "GaitVisualization.h"
#include "GaitClassifier.h"
#include "PersonIdentifier.h"
#include "PathConfig.h"
#include "BatchProcessor.h"
#include "GaitUtils.h"
#include "FeatureHandler.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <numeric>

// Helper struct to hold frame processing results
struct FrameProcessingResult {
    std::vector<double> features;
    std::string filename;
    cv::Mat symmetryMap;
    cv::Mat originalFrame;
};

void processSequenceParallel(
    const std::vector<cv::Mat>& frames,
    const std::vector<std::string>& filenames, 
    gait::GaitAnalyzer& analyzer,
    std::vector<std::pair<std::vector<double>, std::string>>& sequenceFeatures,
    bool showVisualization) {
    
    const size_t numThreads = std::thread::hardware_concurrency();
    const size_t windowSize = 30;  // Process 30 frames at a time
    const size_t windowStride = 15; // Overlap windows by 50%
    std::mutex featuresMutex;
    std::mutex visualizationMutex;
    
    // Process overlapping windows of frames
    std::vector<std::future<std::vector<FrameProcessingResult>>> futures;
    
    for (size_t windowStart = 0; windowStart + windowSize <= frames.size(); 
         windowStart += windowStride) {
        
        futures.push_back(std::async(std::launch::async, 
            [&analyzer, &frames, &filenames, windowStart, windowSize]() {
                std::vector<FrameProcessingResult> windowResults;
                std::vector<std::vector<double>> windowFeatures;
                
                for (size_t i = windowStart; i < windowStart + windowSize && i < frames.size(); i++) {
                    FrameProcessingResult result;
                    result.originalFrame = frames[i].clone();
                    result.filename = filenames[i];
                    result.symmetryMap = analyzer.processFrame(frames[i]);
                    result.features = analyzer.extractGaitFeatures(result.symmetryMap);
                    windowResults.push_back(std::move(result));
                    
                    if (!result.features.empty()) {
                        windowFeatures.push_back(result.features);
                    }
                }
                
                return windowResults;
            }));
    }
    
    // Process results as they complete
    for (auto& future : futures) {
        auto windowResults = future.get();
        if (!windowResults.empty()) {
            std::lock_guard<std::mutex> lock(featuresMutex);
            
            // Get normalized features for this window
            std::vector<std::vector<double>> windowFeatures;
            for (const auto& result : windowResults) {
                if (!result.features.empty()) {
                    windowFeatures.push_back(result.features);
                }
            }
            
            if (!windowFeatures.empty()) {
                std::vector<double> normalizedFeatures = 
                    gait::FeatureHandler::normalizeAndResampleFeatures(windowFeatures);
                // Store features with the filename from first frame in window
                sequenceFeatures.emplace_back(normalizedFeatures, windowResults[0].filename);
            }
            
            if (showVisualization) {
                std::lock_guard<std::mutex> visLock(visualizationMutex);
                gait::visualization::displayResults(
                    windowResults[0].originalFrame,
                    windowResults[0].symmetryMap,
                    windowResults[0].features
                );
            }
        }
    }
}

int main() {
    try {
        // Initialize components with timing measurements
        auto startTime = std::chrono::steady_clock::now();
        
        // Initialize configuration
        auto& config = gait::PathConfig::getInstance();
        if (!config.loadConfig("")) {
            std::cerr << "Failed to initialize path configuration" << std::endl;
            return 1;
        }

        // Initialize components with enhanced parameters
        gait::Loader loader(config.getPath("DATASET_ROOT"));
        gait::SymmetryParams analyzerParams(27.0, 90.0, 0.1);
        gait::GaitAnalyzer analyzer(analyzerParams);

        // Initialize classifier with optimized parameters
        gait::ClassifierParams classifierParams(
            0.65,   // minConfidenceThreshold - increased from 0.65
            5,      // kNeighbors - increased from 5
            100.0,   // maxValidDistance - decreased from 100.0
            0.5,    // temporalWeight
            0.5     // spatialWeight - increased weight of spatial features
        );
        gait::GaitClassifier classifier(classifierParams);
        
        auto initTime = std::chrono::steady_clock::now();
        std::cout << "Initialization time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         initTime - startTime).count() << "ms\n";

        // Handle visualization setup
        bool showVisualization = false;
        std::cout << "Show visualization? (y/n): ";
        std::string input;
        std::getline(std::cin, input);
        showVisualization = (input == "y");

        if (showVisualization && !gait::visualization::initializeWindows()) {
            std::cerr << "Failed to initialize visualization windows" << std::endl;
            return 1;
        }

        // Load and process all subjects
        std::cout << "\nLoading subject data...\n";
        auto loadStart = std::chrono::steady_clock::now();
        
        // Updated data structure to store features with filenames
        std::map<std::string, std::vector<std::pair<std::vector<double>, std::string>>> personFeatures;
        auto allSubjectData = loader.loadAllSubjectsWithFilenames(true);
        
        auto loadEnd = std::chrono::steady_clock::now();
        std::cout << "Data loading time: " 
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         loadEnd - loadStart).count() << "s\n";

        // Process subjects with progress tracking
        std::cout << "\nProcessing subjects...\n";
        size_t totalSubjects = allSubjectData.size();
        size_t processedSubjects = 0;

        for (const auto& [subjectId, data] : allSubjectData) {
            std::vector<std::pair<std::vector<double>, std::string>> sequenceFeatures;
            processSequenceParallel(data.frames, data.filenames, analyzer, sequenceFeatures, showVisualization);
            
            if (!sequenceFeatures.empty()) {
                personFeatures[subjectId] = sequenceFeatures;
            }

            // Update progress
            processedSubjects++;
            float progress = (float)processedSubjects / totalSubjects * 100;
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                      << progress << "%" << std::flush;
        }
        std::cout << "\nProcessing complete!\n";

        // Train classifier
        if (!personFeatures.empty()) {
            std::cout << "\nTraining classifier...\n";
            if (classifier.analyzePatterns(personFeatures)) {
                std::cout << "Classifier training complete.\n";
                
                // Test classification on training data
                for (const auto& [person, features_and_filenames] : personFeatures) {
                    if (!features_and_filenames.empty()) {
                        const auto& [features, filename] = features_and_filenames[0];
                        
                        auto [predictedPerson, confidence] = 
                            classifier.identifyPerson(features, filename);
                            
                        std::cout << "Person " << person 
                                << " (file: " << filename << ")"
                                << " identified as: " << predictedPerson 
                                << " (confidence: " << std::fixed 
                                << std::setprecision(4) << confidence << ")\n";
                    }
                }
            }
        }

        // Interactive mode remains the same...

        if (showVisualization) {
            gait::visualization::cleanupWindows();
        }

        auto endTime = std::chrono::steady_clock::now();
        std::cout << "\nTotal execution time: " 
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         endTime - startTime).count() << "s\n";
        bool continueRunning = true;
        while (continueRunning) {
            std::cout << "\nGait Analysis Options:\n"
                    << "1. Analyze single image\n"
                    << "2. Analyze folder\n"
                    << "3. Exit\n"
                    << "Choose option (1-3): ";
            
            std::string choice;
            std::getline(std::cin, choice);

            if (choice == "1") {
                std::cout << "Enter image path: ";
                std::string imagePath;
                std::getline(std::cin, imagePath);

                gait::PersonIdentifier identifier(analyzer, classifier);
                try {
                    auto [personId, confidence] = identifier.identifyFromImage(imagePath, showVisualization);
                    std::cout << "\nAnalysis Results:\n"
                            << "Identified Person: " << personId << "\n"
                            << "Confidence: " << std::fixed << std::setprecision(4) 
                            << confidence << "\n";
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing image: " << e.what() << std::endl;
                }
            }
            else if (choice == "2") {
                std::cout << "Enter folder path: ";
                std::string folderPath;
                std::getline(std::cin, folderPath);

                gait::BatchProcessor batchProcessor(analyzer, classifier);
                try {
                    std::cout << "\nProcessing folder...\n";
                    auto results = batchProcessor.processDirectory(folderPath, showVisualization);
                    
                    if (results.empty()) {
                        std::cout << "No valid images found in directory.\n";
                    }
                    else {
                        std::cout << "\nProcessed " << results.size() << " files.\n";
                        
                        // Group results by predicted person
                        std::map<std::string, std::vector<double>> confidences;
                        for (const auto& result : results) {
                            confidences[result.predictedPerson].push_back(result.confidence);
                        }

                        // Print summary for each person
                        for (const auto& [person, confs] : confidences) {
                            double avgConf = std::accumulate(confs.begin(), confs.end(), 0.0) / confs.size();
                            double percentage = (100.0 * confs.size()) / results.size();
                            
                            std::cout << person << ": " 
                                    << confs.size() << " images (" 
                                    << std::fixed << std::setprecision(1) << percentage << "%) "
                                    << "avg conf: " << std::setprecision(4) << avgConf << "\n";
                        }
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing folder: " << e.what() << std::endl;
                }
            }
            else if (choice == "3") {
                continueRunning = false;
                std::cout << "Exiting...\n";
            }
            else {
                std::cout << "Invalid choice. Please try again.\n";
            }

            std::cin.clear();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
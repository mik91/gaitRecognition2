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
        auto startTime = std::chrono::steady_clock::now();
        
        // Initialize configuration
        auto& config = gait::PathConfig::getInstance();
        if (!config.loadConfig("")) {
            std::cerr << "Failed to initialize path configuration" << std::endl;
            return 1;
        }

        // Initialize components
        gait::SymmetryParams analyzerParams(27.0, 90.0, 0.1);
        gait::GaitAnalyzer analyzer(analyzerParams);
        gait::ClassifierParams classifierParams(0.65, 5, 100.0, 0.5, 0.5);
        gait::GaitClassifier classifier(classifierParams);

        // Ask user whether to train new model or load existing one
        std::cout << "Choose operation mode:\n"
                  << "1. Train new model\n"
                  << "2. Load existing model\n"
                  << "Enter choice (1-2): ";
        
        std::string modeChoice;
        std::getline(std::cin, modeChoice);

        if (modeChoice == "1") {
            // Training mode
            gait::Loader loader(config.getPath("DATASET_ROOT"));
            
            std::cout << "\nLoading subject data...\n";
            auto loadStart = std::chrono::steady_clock::now();
            
            auto allSubjectData = loader.loadAllSubjectsWithFilenames(true);
            
            auto loadEnd = std::chrono::steady_clock::now();
            std::cout << "Data loading time: " 
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             loadEnd - loadStart).count() << "s\n";

            // Process and train as before...
            std::map<std::string, std::vector<std::pair<std::vector<double>, 
                    std::string>>> personFeatures;
            size_t totalSubjects = allSubjectData.size();
            size_t processedSubjects = 0;

            loadStart = std::chrono::steady_clock::now();

            for (const auto& [subjectId, data] : allSubjectData) {
                std::vector<std::pair<std::vector<double>, std::string>> sequenceFeatures;
                processSequenceParallel(data.frames, data.filenames, analyzer, 
                                      sequenceFeatures, false);
                
                if (!sequenceFeatures.empty()) {
                    personFeatures[subjectId] = sequenceFeatures;
                }

                processedSubjects++;
                float progress = (float)processedSubjects / totalSubjects * 100;
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                          << progress << "%" << std::flush;
            }
            
            if (!personFeatures.empty()) {
                std::cout << "\nTraining classifier...\n";
                if (classifier.analyzePatterns(personFeatures)) {
                    // Details about the classifier
                    // Validate the model
                    std::cout << "\nValidating classifier...\n";
                    for (const auto& [subject, features] : personFeatures) {
                        if (!features.empty()) {
                            auto [predicted, confidence] = classifier.identifyPerson(
                                features[0].first, features[0].second);
                            std::cout << "Subject " << subject << " -> predicted as " 
                                    << predicted << " (confidence: " << confidence << ")\n";
                        }
                    }
                    std::cout << "Classifier training complete.\n";
                    classifier.saveModel(config.getPath("RESULTS_DIR") + "/gait_classifier.yml");
                }
            }

            loadEnd = std::chrono::steady_clock::now();

            std::cout << "Training time: " 
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             loadEnd - loadStart).count() << "s\n";
        }
        else if (modeChoice == "2") {
            // Load existing model
            std::string modelPath = config.getPath("RESULTS_DIR") + "/gait_classifier.yml";
            if (!classifier.loadModel(modelPath)) {
                std::cerr << "Failed to load model from " << modelPath << std::endl;
                return 1;
            }
            std::cout << "Model loaded successfully.\n";
        }
        else {
            std::cerr << "Invalid choice\n";
            return 1;
        }

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

        // Interactive mode
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
                    auto [personId, confidence] = identifier.identifyFromImage(
                        imagePath, showVisualization);
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
        }

        if (showVisualization) {
            gait::visualization::cleanupWindows();
        }

        auto endTime = std::chrono::steady_clock::now();
        std::cout << "\nTotal execution time: " 
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         endTime - startTime).count() << "s\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
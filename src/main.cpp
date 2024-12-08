#include "Loader.h"
#include "GaitAnalyzer.h"
#include "GaitVisualization.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <GaitClassifier.h>
#include <PersonIdentifier.h>
#include <PathConfig.h>

std::vector<double> accumulateSequenceFeatures(const std::vector<std::vector<double>>& frameFeatures) {
    if (frameFeatures.empty()) {
        return std::vector<double>();
    }
    
    size_t featureSize = frameFeatures[0].size();
    
    // Verify all vectors have the same size
    for (const auto& features : frameFeatures) {
        if (features.size() != featureSize) {
            std::cerr << "Warning: Inconsistent feature vector sizes. Expected: " 
                     << featureSize << ", Got: " << features.size() << std::endl;
            return std::vector<double>();
        }
    }
    
    std::vector<double> meanFeatures(featureSize, 0.0);
    std::vector<double> varFeatures(featureSize, 0.0);
    
    // Safely calculate mean
    for (const auto& frame : frameFeatures) {
        for (size_t i = 0; i < featureSize; i++) {
            meanFeatures[i] += frame[i];
        }
    }
    
    for (auto& val : meanFeatures) {
        val /= frameFeatures.size();
    }
    
    // Safely calculate variance
    for (const auto& frame : frameFeatures) {
        for (size_t i = 0; i < featureSize; i++) {
            double diff = frame[i] - meanFeatures[i];
            varFeatures[i] += diff * diff;
        }
    }
    
    for (auto& val : varFeatures) {
        val = std::sqrt(val / frameFeatures.size());
    }
    
    // Print debug information
    std::cout << "Frame feature statistics:\n"
              << "Number of frames: " << frameFeatures.size() << "\n"
              << "Features per frame: " << featureSize << "\n"
              << "Sample of mean values (first 5):\n";
    
    for (size_t i = 0; i < std::min(size_t(5), featureSize); i++) {
        std::cout << "Feature " << i << ": mean=" << meanFeatures[i] 
                 << ", stddev=" << varFeatures[i] << "\n";
    }
    
    // Combine mean and variance features with bounds checking
    std::vector<double> combinedFeatures;
    combinedFeatures.reserve(featureSize * 2);
    
    for (size_t i = 0; i < featureSize; i++) {
        // Only include features with meaningful variation
        if (varFeatures[i] > 1e-10) {
            combinedFeatures.push_back(meanFeatures[i]);
            combinedFeatures.push_back(varFeatures[i]);
        }
    }
    
    std::cout << "Combined feature vector size: " << combinedFeatures.size() << "\n";
    
    return combinedFeatures;
}

void processSequence(const std::string& seq, const std::vector<cv::Mat>& frames, 
                    gait::GaitAnalyzer& analyzer,
                    std::map<std::string, std::vector<std::vector<double>>>& personFeatures,
                    const std::string& personId) {
    if (frames.empty()) {
        std::cout << "Warning: No frames found for " << personId << " sequence " << seq << std::endl;
        return;
    }

    std::cout << "Processing " << personId << " sequence " << seq 
              << " (" << frames.size() << " frames)" << std::endl;
    
    std::vector<std::vector<double>> sequenceFrameFeatures;
    const size_t EXPECTED_FEATURE_SIZE = 124;  // Fixed size based on our feature extraction
    
    // Process each frame
    for (const auto& frame : frames) {
        cv::Mat symmetryMap = analyzer.processFrame(frame);
        std::vector<double> frameFeatures = analyzer.extractGaitFeatures(symmetryMap);
        
        // Ensure consistent feature size
        if (frameFeatures.size() > 0) {
            if (frameFeatures.size() > EXPECTED_FEATURE_SIZE) {
                frameFeatures.resize(EXPECTED_FEATURE_SIZE);
            } else if (frameFeatures.size() < EXPECTED_FEATURE_SIZE) {
                frameFeatures.resize(EXPECTED_FEATURE_SIZE, 0.0);  // Pad with zeros
            }
            sequenceFrameFeatures.push_back(frameFeatures);
        }
    }
    
    if (!sequenceFrameFeatures.empty()) {
        // Average features across frames
        std::vector<double> avgFeatures(EXPECTED_FEATURE_SIZE, 0.0);
        for (const auto& frameFeatures : sequenceFrameFeatures) {
            for (size_t i = 0; i < EXPECTED_FEATURE_SIZE; i++) {
                avgFeatures[i] += frameFeatures[i];
            }
        }
        
        for (auto& val : avgFeatures) {
            val /= sequenceFrameFeatures.size();
        }
        
        // Add features to person's data
        personFeatures[personId].push_back(avgFeatures);
        
        std::cout << "Added features for " << personId << " sequence " << seq << ":\n"
                 << "Feature vector size: " << avgFeatures.size() << std::endl;
    }
}

int main() {
    try {
        // Initialize components
        auto& config = gait::PathConfig::getInstance();

        // Get the executable path
        std::filesystem::path execPath = std::filesystem::canonical("/proc/self/exe");
        std::filesystem::path projectRoot = execPath.parent_path().parent_path();
        std::filesystem::path configPath = projectRoot / "config" / "paths.conf";

        if (!config.loadConfig(configPath.string())) {
            std::cerr << "Failed to load path configuration from: " << configPath << std::endl;
            return 1;
        }

        gait::Loader loader(config.getPath("DATASET_ROOT"));
        
        gait::SymmetryParams params(27.0, 90.0, 0.1);
        gait::GaitAnalyzer analyzer(params);
        gait::GaitClassifier classifier;

        // Collect features per person
        std::map<std::string, std::vector<std::vector<double>>> personFeatures;
        std::vector<std::string> people = {"test", "test2"};
        std::vector<std::string> conditions = {"nm", "bg"};

        bool showVisualization = false;
        std::string input;

        while (true) {
            std::cout << "Show visualization? (y/n): ";
            std::getline(std::cin, input);

            if (input == "y") {
                showVisualization = true;
                break;
            } else if (input == "n") {
                showVisualization = false;
                break;
            }

        }

        if(showVisualization) {
            // Create visualization windows
            if (!gait::visualization::initializeWindows()) {
                std::cerr << "Failed to initialize visualization windows" << std::endl;
                return 1;
            }

            // Create additional windows for detailed features
            cv::namedWindow("Detailed Features", cv::WINDOW_NORMAL);
            cv::namedWindow("Regional Features", cv::WINDOW_NORMAL);
            cv::namedWindow("Temporal Features", cv::WINDOW_NORMAL);

            // Set initial sizes for better visibility
            cv::resizeWindow("Detailed Features", 800, 400);
            cv::resizeWindow("Regional Features", 400, 400);
            cv::resizeWindow("Temporal Features", 600, 300);
        }


        // Process each person's sequences
        for (const auto& person : people) {
            for (const auto& condition : conditions) {
                auto frames = loader.loadSequence(person, condition, 1);
                if (!frames.empty()) {
                    processSequence(condition, frames, analyzer, personFeatures, person);
                    
                    // Show processing progress
                    if (showVisualization) {
                        for (const auto& frame : frames) {
                            // Process frame
                            cv::Mat symmetryMap = analyzer.processFrame(frame);
                            std::vector<double> frameFeatures = analyzer.extractGaitFeatures(symmetryMap);
                            
                            // Use the comprehensive display function
                            bool continueProcessing = gait::visualization::displayResults(
                                frame,                  // original frame
                                symmetryMap,           // symmetry map
                                frameFeatures         // extracted features
                            );
                            
                            // Also show detailed feature visualizations
                            cv::Mat featureVis = gait::visualization::visualizeGaitFeatures(frameFeatures);
                            if (!featureVis.empty()) {
                                cv::imshow("Detailed Features", featureVis);
                            }
                            
                            cv::Mat regionalVis = gait::visualization::visualizeRegionalFeatures(
                                std::vector<double>(frameFeatures.begin(), frameFeatures.begin() + 4)
                            );
                            if (!regionalVis.empty()) {
                                cv::imshow("Regional Features", regionalVis);
                            }
                            
                            cv::Mat temporalVis = gait::visualization::visualizeTemporalFeatures(
                                std::vector<double>(frameFeatures.begin() + 4, frameFeatures.begin() + 7)
                            );
                            if (!temporalVis.empty()) {
                                cv::imshow("Temporal Features", temporalVis);
                            }
                            
                            // Handle window layout
                            cv::moveWindow("Original Frame", 0, 0);
                            cv::moveWindow("Symmetry Map", 650, 0);
                            cv::moveWindow("Detailed Features", 1300, 0);
                            cv::moveWindow("Regional Features", 0, 500);
                            cv::moveWindow("Temporal Features", 650, 500);
                            
                            char key = cv::waitKey(30);
                            if (key == 27) {  // ESC key
                                return 0;
                            }
                        }
                    }
                }
            }
        }

        // Train classifier if we have data
        bool hasData = false;
        for (const auto& [person, sequences] : personFeatures) {
            if (!sequences.empty()) {
                hasData = true;
                break;
            }
        }

        if (hasData) {
            std::cout << "\nTraining classifier with available data..." << std::endl;
            if (classifier.analyzePatterns(personFeatures, showVisualization)) {
                // Test each sequence
                for (const auto& [person, sequences] : personFeatures) {
                    for (const auto& sequence : sequences) {
                        auto [predictedPerson, confidence] = classifier.identifyPerson(sequence, showVisualization);
                        std::cout << "Sequence from " << person 
                                << " identified as: " << predictedPerson 
                                << " (confidence: " << confidence << ")\n";
                    }
                }
            }
        } else {
            std::cout << "No valid data found for training" << std::endl;
        }
        
        if (classifier.isModelTrained()) {
            gait::PersonIdentifier identifier(analyzer, classifier);
        
            while (true) {
                std::cout << "\nEnter path to image for identification (or 'quit' to exit): ";
                std::getline(std::cin, input);
                
                if (input == "quit") break;
                
                try {
                    auto [predictedPerson, confidence] = identifier.identifyFromImage(input, showVisualization);
                    std::cout << "Predicted person: " << predictedPerson << "\n"
                            << "Confidence: " << confidence << "\n";
                } catch (const std::exception& e) {
                    std::cerr << "Error processing image: " << e.what() << "\n";
                }
            }
        }

        cv::waitKey(0);
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
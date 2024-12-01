#include "Loader.h"
// #include "GaitAnalyzer.h"
// #include "GaitClassifier.h"
#include <iostream>
// #include <opencv2/highgui.hpp>

int main() {
    // try {
    //     // Initialize loader with dataset path
    //     #ifdef _WIN32
    //         gait::Loader loader("C:\\Users\\kamar\\OneDrive\\Documents\\UDEM\\Session 1\\IFT6150\\gaitRecognition2\\data\\CASIA_B");
    //     #else
    //         gait::Loader loader("/u/kamarami/Documents/linux-gaitanalyzer/data/CASIA_B");
    //     #endif

    //     // Initialize GaitAnalyzer with default parameters
    //     gait::SymmetryParams params(27.0, 90.0, 0.1); // sigma, mu, threshold
    //     gait::GaitAnalysisSystem system(params);

    //     // Print available subjects
    //     std::cout << "Available subjects:" << std::endl;
    //     for (const auto& subject : loader.getSubjectIds()) {
    //         std::cout << " - " << subject << std::endl;
    //     }

    //     // Load training data
    //     std::vector<std::vector<cv::Mat>> normalSequences;
    //     std::vector<std::vector<cv::Mat>> abnormalSequences;

    //     // Load normal gait sequences (e.g., from 'nm' condition)
    //     try {
    //         auto frames = loader.loadSequence("test", "nm", 1);
    //         if (!frames.empty()) {
    //             normalSequences.push_back(frames);
    //         }
    //     } catch (const std::exception& e) {
    //         std::cerr << "Error loading normal sequence: " << e.what() << std::endl;
    //     }

    //     // Load potentially abnormal gait sequences (e.g., from 'bg' condition)
    //     try {
    //         auto frames = loader.loadSequence("test", "bg", 1);
    //         if (!frames.empty()) {
    //             abnormalSequences.push_back(frames);
    //         }
    //     } catch (const std::exception& e) {
    //         std::cerr << "Error loading abnormal sequence: " << e.what() << std::endl;
    //     }

    //     // Train the classifier if we have both types of sequences
    //     if (!normalSequences.empty() && !abnormalSequences.empty()) {
    //         std::cout << "Training classifier..." << std::endl;
    //         if (system.trainClassifier(normalSequences, abnormalSequences)) {
    //             std::cout << "Classifier trained successfully" << std::endl;
    //         }
    //     }

    //     // Process and visualize a test sequence
    //     try {
    //         // Load a test sequence
    //         auto testFrames = loader.loadSequence("test", "nm", 2);
            
    //         if (!testFrames.empty()) {
    //             std::cout << "Processing test sequence..." << std::endl;
                
    //             // Process each frame
    //             for (size_t i = 0; i < testFrames.size(); i++) {
    //                 const auto& frame = testFrames[i];
                    
    //                 // Create windows for visualization
    //                 cv::namedWindow("Original Frame", cv::WINDOW_NORMAL);
    //                 cv::namedWindow("Symmetry Map", cv::WINDOW_NORMAL);
    //                 cv::namedWindow("Features", cv::WINDOW_NORMAL);
                    
    //                 // Process frame and get symmetry map
    //                 cv::Mat symmetryMap = system.getAnalyzer().processFrame(frame);
                    
    //                 // Extract features
    //                 std::vector<double> features = system.getAnalyzer().extractGaitFeatures(symmetryMap);
                    
    //                 // Visualize results
    //                 cv::imshow("Original Frame", frame);
    //                 cv::imshow("Symmetry Map", 
    //                          gait::visualization::visualizeSymmetryMap(symmetryMap));
    //                 cv::imshow("Features", 
    //                          gait::visualization::visualizeGaitFeatures(features));
                    
    //                 // Display frame number
    //                 std::cout << "Processing frame " << i + 1 << "/" 
    //                          << testFrames.size() << std::endl;
                    
    //                 // Wait for key press (use ESC to exit)
    //                 char key = cv::waitKey(30);
    //                 if (key == 27) break; // ESC key
    //             }
                
    //             // If classifier is trained, perform classification
    //             if (!normalSequences.empty() && !abnormalSequences.empty()) {
    //                 double confidence;
    //                 auto gaitType = system.analyzeSequence(testFrames, confidence);
                    
    //                 std::cout << "Classification result: " 
    //                          << (gaitType == gait::GaitClassifier::GaitType::NORMAL ? 
    //                              "Normal" : "Abnormal")
    //                          << " (confidence: " << confidence << ")" << std::endl;
    //             }
    //         }
    //     } catch (const std::exception& e) {
    //         std::cerr << "Error processing test sequence: " << e.what() << std::endl;
    //     }

    //     // Clean up
    //     cv::destroyAllWindows();
        
    // } catch (const std::exception& e) {
    //     std::cerr << "Error initializing loader: " << e.what() << std::endl;
    //     return 1;
    // }
    
    return 0;
}
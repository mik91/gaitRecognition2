#include "PersonIdentifier.h"
#include "GaitVisualization.h"
#include <stdexcept>

namespace gait {

PersonIdentifier::PersonIdentifier(GaitAnalyzer& analyzer, GaitClassifier& classifier)
    : analyzer_(analyzer), classifier_(classifier) {}

std::pair<std::string, double> PersonIdentifier::identifyFromImage(const std::string& imagePath) {
    // Load and validate image
    cv::Mat inputImage = cv::imread(imagePath);
    if (inputImage.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }

    // Process image through GaitAnalyzer
    cv::Mat symmetryMap = analyzer_.processFrame(inputImage);
    std::vector<double> features = analyzer_.extractGaitFeatures(symmetryMap);

    // Use classifier to identify person
    auto [personId, confidence] = classifier_.identifyPerson(features);

    // Visualize results
    cv::namedWindow("Input Image", cv::WINDOW_NORMAL);
    cv::namedWindow("Symmetry Map", cv::WINDOW_NORMAL);
    cv::imshow("Input Image", inputImage);
    cv::imshow("Symmetry Map", visualization::visualizeSymmetryMap(symmetryMap));
    
    // Display prediction text
    cv::Mat resultDisplay = inputImage.clone();
    std::string resultText = "Predicted: " + personId;
    std::string confidenceText = "Confidence: " + std::to_string(confidence);
    cv::putText(resultDisplay, resultText, cv::Point(20, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::putText(resultDisplay, confidenceText, cv::Point(20, 70), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Result", resultDisplay);
    
    cv::waitKey(0);
    return {personId, confidence};
}

} // namespace gait
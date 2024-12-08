#pragma once

#include "GaitAnalyzer.h"
#include "GaitClassifier.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>

namespace gait {

class PersonIdentifier {
public:
    // Constructor
    PersonIdentifier(GaitAnalyzer& analyzer, GaitClassifier& classifier);

    // Main identification method
    std::pair<std::string, double> identifyFromImage(const std::string& imagePath, 
                                    bool visualize = false);

private:
    GaitAnalyzer& analyzer_;
    GaitClassifier& classifier_;
};

} // namespace gait
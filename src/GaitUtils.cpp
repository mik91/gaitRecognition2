// GaitUtils.cpp
#include "GaitUtils.h"
#include <iostream>
#include <cmath>

namespace gait {
namespace utils {

std::vector<double> accumulateSequenceFeatures(const std::vector<std::vector<double>>& frameFeatures) {
    if (frameFeatures.empty()) {
        return std::vector<double>();
    }
    
    size_t featureSize = frameFeatures[0].size();
    std::vector<double> meanFeatures(featureSize, 0.0);
    std::vector<double> varFeatures(featureSize, 0.0);
    
    // Calculate mean
    for (const auto& frame : frameFeatures) {
        for (size_t i = 0; i < featureSize; i++) {
            meanFeatures[i] += frame[i];
        }
    }
    
    for (auto& val : meanFeatures) {
        val /= frameFeatures.size();
    }
    
    // Calculate variance
    for (const auto& frame : frameFeatures) {
        for (size_t i = 0; i < featureSize; i++) {
            double diff = frame[i] - meanFeatures[i];
            varFeatures[i] += diff * diff;
        }
    }
    
    for (auto& val : varFeatures) {
        val = std::sqrt(val / frameFeatures.size());
    }
    
    // Combine mean and variance features
    std::vector<double> combinedFeatures;
    combinedFeatures.reserve(featureSize * 2);
    
    for (size_t i = 0; i < featureSize; i++) {
        if (varFeatures[i] > 1e-10) {
            combinedFeatures.push_back(meanFeatures[i]);
            combinedFeatures.push_back(varFeatures[i]);
        }
    }
    
    return combinedFeatures;
}

} // namespace utils
} // namespace gait
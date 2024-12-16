// FeatureHandler.cpp
#include "FeatureHandler.h"
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace gait {

std::vector<double> FeatureHandler::normalizeAndResampleFeatures(
    const std::vector<std::vector<double>>& frameFeatures,
    size_t targetLength) {
    
    if (frameFeatures.empty()) {
        return std::vector<double>();
    }

    targetLength = frameFeatures[0].size();
    
    std::vector<double> meanFeatures(targetLength, 0.0);
    std::vector<double> stdFeatures(targetLength, 0.0);
    
    // compute means
    for (const auto& frame : frameFeatures) {
        for (size_t i = 0; i < targetLength; i++) {
            meanFeatures[i] += frame[i];
        }
    }
    
    for (auto& val : meanFeatures) {
        val /= frameFeatures.size();
    }
    
    // compute std dev
    for (const auto& frame : frameFeatures) {
        for (size_t i = 0; i < targetLength; i++) {
            double diff = frame[i] - meanFeatures[i];
            stdFeatures[i] += diff * diff;
        }
    }
    
    // Combine features
    std::vector<double> combined;
    combined.reserve(targetLength * 2);
    
    for (size_t i = 0; i < targetLength; i++) {
        combined.push_back(meanFeatures[i]);
        combined.push_back(std::sqrt(stdFeatures[i] / frameFeatures.size()));
    }
    
    return combined;
}
} // namespace gait
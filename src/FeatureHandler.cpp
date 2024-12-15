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

    // Don't resample, keep original feature size
    targetLength = frameFeatures[0].size();
    
    // Compute mean features
    std::vector<double> meanFeatures(targetLength, 0.0);
    std::vector<double> stdFeatures(targetLength, 0.0);
    
    // First pass: compute means
    for (const auto& frame : frameFeatures) {
        for (size_t i = 0; i < targetLength; i++) {
            meanFeatures[i] += frame[i];
        }
    }
    
    for (auto& val : meanFeatures) {
        val /= frameFeatures.size();
    }
    
    // Second pass: compute std dev
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

std::vector<double> FeatureHandler::interpolateFeatures(
    const std::vector<double>& features,
    size_t targetLength) {
    
    if (features.empty() || targetLength == 0) {
        return std::vector<double>();
    }

    if (features.size() == targetLength) {
        return features;
    }

    std::vector<double> result(targetLength);
    double scale = static_cast<double>(features.size() - 1) / (targetLength - 1);

    for (size_t i = 0; i < targetLength; ++i) {
        double pos = i * scale;
        size_t idx = static_cast<size_t>(pos);
        double frac = pos - idx;

        if (idx + 1 < features.size()) {
            // Linear interpolation
            result[i] = features[idx] * (1 - frac) + features[idx + 1] * frac;
        } else {
            // Handle edge case for last element
            result[i] = features[features.size() - 1];
        }
    }

    return result;
}

std::vector<double> FeatureHandler::computeStatistics(
    const std::vector<std::vector<double>>& features) {
    
    if (features.empty() || features[0].empty()) {
        return std::vector<double>();
    }

    size_t numFrames = features.size();
    size_t numFeatures = features[0].size();
    
    // Initialize statistics vectors
    std::vector<double> means(numFeatures, 0.0);
    std::vector<double> variances(numFeatures, 0.0);
    std::vector<double> medians(numFeatures, 0.0);
    std::vector<double> ranges(numFeatures, 0.0);
    
    // Temporary storage for computing medians and ranges
    std::vector<std::vector<double>> featureValues(numFeatures);
    for (size_t i = 0; i < numFeatures; ++i) {
        featureValues[i].reserve(numFrames);
    }
    
    // First pass: compute means and collect values
    for (const auto& frame : features) {
        for (size_t i = 0; i < numFeatures; ++i) {
            means[i] += frame[i];
            featureValues[i].push_back(frame[i]);
        }
    }
    
    // Finalize means
    for (double& mean : means) {
        mean /= numFrames;
    }
    
    // Second pass: compute variances
    for (const auto& frame : features) {
        for (size_t i = 0; i < numFeatures; ++i) {
            double diff = frame[i] - means[i];
            variances[i] += diff * diff;
        }
    }
    
    // Finalize variances and compute additional statistics
    for (size_t i = 0; i < numFeatures; ++i) {
        // Standard deviation
        variances[i] = std::sqrt(variances[i] / numFrames);
        
        // Sort values for median and range
        std::sort(featureValues[i].begin(), featureValues[i].end());
        medians[i] = featureValues[i][numFrames / 2];
        ranges[i] = featureValues[i].back() - featureValues[i].front();
    }
    
    // Combine all statistics
    std::vector<double> combined;
    combined.reserve(numFeatures * 4);  // means, variances, medians, ranges
    
    combined.insert(combined.end(), means.begin(), means.end());
    combined.insert(combined.end(), variances.begin(), variances.end());
    combined.insert(combined.end(), medians.begin(), medians.end());
    combined.insert(combined.end(), ranges.begin(), ranges.end());
    
    return combined;
}

} // namespace gait
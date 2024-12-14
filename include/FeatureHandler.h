// FeatureHandler.h
#pragma once
#include <vector>
#include <cstddef>

namespace gait {

/**
 * @brief Handles feature processing and normalization for gait recognition
 * 
 * This class provides static methods for processing, normalizing, and 
 * resampling feature vectors to ensure consistent lengths and proper
 * statistical representation of gait features.
 */
class FeatureHandler {
public:
    /**
     * @brief Normalizes and resamples frame features to a target length
     * 
     * @param frameFeatures Vector of feature vectors from multiple frames
     * @param targetLength Desired length of output feature vector (default 210)
     * @return Normalized and resampled feature vector
     */
    static std::vector<double> normalizeAndResampleFeatures(
        const std::vector<std::vector<double>>& frameFeatures,
        size_t targetLength = 210
    );

private:
    /**
     * @brief Interpolates a feature vector to a target length
     */
    static std::vector<double> interpolateFeatures(
        const std::vector<double>& features,
        size_t targetLength
    );
    
    /**
     * @brief Computes statistical measures from frame features
     */
    static std::vector<double> computeStatistics(
        const std::vector<std::vector<double>>& features
    );
};

} // namespace gait
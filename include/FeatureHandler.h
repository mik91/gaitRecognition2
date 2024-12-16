#pragma once
#include <vector>
#include <cstddef>

namespace gait {
class FeatureHandler {
public:
    static std::vector<double> normalizeAndResampleFeatures(
        const std::vector<std::vector<double>>& frameFeatures,
        size_t targetLength = 210
    );
};

} // namespace gait
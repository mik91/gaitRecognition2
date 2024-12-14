// GaitUtils.h
#pragma once
#include <vector>
#include <string>

namespace gait {
namespace utils {

// Function declaration
std::vector<double> accumulateSequenceFeatures(const std::vector<std::vector<double>>& frameFeatures);

} // namespace utils
} // namespace gait
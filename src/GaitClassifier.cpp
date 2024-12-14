// GaitClassifier.cpp
#include "GaitClassifier.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cmath>
#include "FeatureHandler.h"
#include <future>

namespace gait {

GaitClassifier::GaitClassifier(const ClassifierParams& params)
    : params_(params), isModelTrained_(false) {}

std::string GaitClassifier::extractCondition(const std::string& filename) const {
    size_t start = filename.find('-') + 1;
    size_t end = filename.find('-', start);
    if (start != std::string::npos && end != std::string::npos) {
        return filename.substr(start, end - start);
    }
    return "unknown";
}

bool GaitClassifier::analyzePatterns(
    const std::map<std::string, 
                  std::vector<std::pair<std::vector<double>, std::string>>>& personFeatures) {
    
    if (personFeatures.empty()) return false;
    
    // Pre-allocate memory for all sequences
    size_t totalSequences = 0;
    for (const auto& [_, sequences] : personFeatures) {
        totalSequences += sequences.size();
    }
    trainingSequences_.reserve(totalSequences);
    trainingData_.reserve(totalSequences);
    trainingLabels_.reserve(totalSequences);

    // Process data in parallel using thread pool
    const size_t numThreads = std::thread::hardware_concurrency();
    std::vector<std::future<std::vector<SequenceInfo>>> futures;
    std::mutex dataMutex;

    for (const auto& [person, sequences] : personFeatures) {
        futures.push_back(std::async(std::launch::async, [this, &person, &sequences]() {
            std::vector<SequenceInfo> threadSequences;
            threadSequences.reserve(sequences.size());

            for (const auto& [seq, filename] : sequences) {
                if (!seq.empty()) {
                    SequenceInfo info;
                    info.label = person;
                    info.condition = extractCondition(filename);
                    info.features = seq;
                    threadSequences.push_back(std::move(info));
                }
            }
            return threadSequences;
        }));
    }

    // Collect results
    for (auto& future : futures) {
        auto threadResults = future.get();
        std::lock_guard<std::mutex> lock(dataMutex);
        trainingSequences_.insert(trainingSequences_.end(), 
                                std::make_move_iterator(threadResults.begin()),
                                std::make_move_iterator(threadResults.end()));
        
        // Also update legacy data structures
        for (const auto& info : threadResults) {
            trainingData_.push_back(info.features);
            trainingLabels_.push_back(info.label);
        }
    }

    if (trainingData_.empty()) return false;

    // Compute statistics in parallel
    std::future<void> statsFuture = std::async(std::launch::async, 
        [this]() { computeFeatureStatistics(); });
    std::future<void> covFuture = std::async(std::launch::async, 
        [this]() { computeCovarianceMatrix(); });

    // Wait for completion
    statsFuture.wait();
    covFuture.wait();
    
    isModelTrained_ = true;
    return true;
}

std::pair<std::string, double> GaitClassifier::identifyPerson(
    const std::vector<double>& testSequence,
    const std::string& testFilename) {
    
    if (!isModelTrained_ || testSequence.empty()) {
        return {"unknown", 0.0};
    }
    
    std::string testCondition = extractCondition(testFilename);
    auto normalizedTest = normalizeFeatures(testSequence);
    
    // Calculate distances in parallel using thread pool
    const size_t numThreads = std::thread::hardware_concurrency();
    const size_t batchSize = trainingSequences_.size() / numThreads;
    std::vector<std::future<std::vector<std::pair<double, SequenceInfo>>>> futures;
    
    for (size_t i = 0; i < numThreads; ++i) {
        size_t start = i * batchSize;
        size_t end = (i == numThreads - 1) ? trainingSequences_.size() 
                                          : (i + 1) * batchSize;
        
        futures.push_back(std::async(std::launch::async, 
            [this, &normalizedTest, start, end]() {
                std::vector<std::pair<double, SequenceInfo>> batchDistances;
                batchDistances.reserve(end - start);
                
                for (size_t j = start; j < end; ++j) {
                    auto normalizedTrain = normalizeFeatures(trainingSequences_[j].features);
                    
                    // Compute distances efficiently
                    double euclidean = computeEuclideanDistance(normalizedTest, normalizedTrain);
                    double dtw = computeDTWDistance(normalizedTest, normalizedTrain);
                    double mahalanobis = computeMahalanobisDistance(normalizedTest, normalizedTrain);
                    
                    double combinedDist = 0.4 * euclidean + 0.4 * dtw + 0.2 * mahalanobis;
                    batchDistances.emplace_back(combinedDist, trainingSequences_[j]);
                }
                return batchDistances;
            }));
    }

    // Collect and merge results
    std::vector<std::pair<double, SequenceInfo>> allDistances;
    allDistances.reserve(trainingSequences_.size());
    
    for (auto& future : futures) {
        auto batchResults = future.get();
        allDistances.insert(allDistances.end(), 
                          std::make_move_iterator(batchResults.begin()),
                          std::make_move_iterator(batchResults.end()));
    }

    // Sort results
    std::partial_sort(allDistances.begin(), 
                     allDistances.begin() + params_.kNeighbors,
                     allDistances.end());

    // Process k-nearest neighbors
    std::map<std::string, double> votes;
    double totalWeight = 0.0;
    
    for (int i = 0; i < params_.kNeighbors; i++) {
        const auto& [dist, info] = allDistances[i];
        double weight = std::exp(-dist);
        
        if (info.condition == testCondition) {
            weight *= 2.0;  // Boost same condition matches
        }
        
        votes[info.label] += weight;
        totalWeight += weight;
    }

    // Find best match
    auto bestMatch = std::max_element(votes.begin(), votes.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    double confidence = computeConditionAwareConfidence(testCondition, allDistances, 
                                                      bestMatch->first);

    if (confidence > params_.minConfidenceThreshold) {
        return {bestMatch->first, confidence};
    }
    
    return {"unknown", confidence};
}

double GaitClassifier::computeConditionAwareConfidence(
    const std::string& testCondition,
    const std::vector<std::pair<double, SequenceInfo>>& distances,
    const std::string& predictedClass) const {
    
    if (distances.empty() || distances.size() < params_.kNeighbors) {
        return 0.0;
    }
    
    // Look at top k and also examine distribution in larger neighborhood
    size_t k = std::min(static_cast<size_t>(params_.kNeighbors), distances.size());
    size_t extendedK = std::min(k * 3, distances.size()); // Look at 3x more neighbors
    
    // Analyze close neighbors
    int matchCount = 0;
    double matchDistance = 0.0;
    double nonMatchDistance = 0.0;
    int nonMatchCount = 0;
    
    // Analyze extended neighborhood
    int extendedMatchCount = 0;
    std::map<std::string, int> classDistribution;
    
    // Process top k neighbors
    for (size_t i = 0; i < k; i++) {
        const auto& [dist, info] = distances[i];
        if (info.label == predictedClass) {
            matchCount++;
            matchDistance += dist;
        } else {
            nonMatchCount++;
            nonMatchDistance += dist;
        }
    }
    
    // Process extended neighborhood
    for (size_t i = 0; i < extendedK; i++) {
        const auto& [dist, info] = distances[i];
        classDistribution[info.label]++;
        if (info.label == predictedClass) {
            extendedMatchCount++;
        }
    }
    
    if (matchCount == 0) return 0.0;
    
    // Calculate basic confidence measures
    matchDistance /= matchCount;
    nonMatchDistance = nonMatchCount > 0 ? nonMatchDistance / nonMatchCount : matchDistance * 2;
    
    double matchRatio = static_cast<double>(matchCount) / k;
    double distanceRatio = matchDistance / nonMatchDistance;
    
    // Calculate distribution score
    double distributionScore = 0.0;
    if (!classDistribution.empty()) {
        int maxCount = 0;
        int totalCount = 0;
        for (const auto& [_, count] : classDistribution) {
            maxCount = std::max(maxCount, count);
            totalCount += count;
        }
        
        // Strong class should dominate the neighborhood
        distributionScore = static_cast<double>(maxCount) / totalCount;
        
        // Penalize if matches are scattered
        if (classDistribution.size() > 2) {  // More than 2 different classes in neighborhood
            distributionScore *= 0.8;
        }
    }
    
    // Calculate extended match ratio
    double extendedMatchRatio = static_cast<double>(extendedMatchCount) / extendedK;
    
    // Combine scores with weights
    double confidence = 0.3 * matchRatio +                 // Close neighbor matches
                       0.2 * (1.0 - distanceRatio) +       // Distance separation
                       0.3 * distributionScore +           // Class distribution
                       0.2 * extendedMatchRatio;          // Extended neighborhood matches
    
    // Scale to [0.5, 1.0] range for valid matches
    confidence = 0.5 + 0.5 * confidence;
    
    // Condition match bonus (smaller than before)
    int conditionMatches = 0;
    for (size_t i = 0; i < k; i++) {
        if (distances[i].second.condition == testCondition && 
            distances[i].second.label == predictedClass) {
            conditionMatches++;
        }
    }
    
    // Small condition match bonus
    confidence += 0.02 * conditionMatches;
    
    // Additional penalty for scattered matches
    if (classDistribution.size() > 2) {
        confidence *= 0.85;  // Penalty for too many different classes
    }
    
    // Cap final confidence
    confidence = std::min(1.0, confidence);
    
    return confidence;
}

double GaitClassifier::computeMahalanobisDistance(
    const std::vector<double>& seq1,
    const std::vector<double>& seq2) const {
    
    if (covarianceMatrix_.empty() || 
        seq1.empty() || seq2.empty() || 
        seq1.size() != seq2.size()) {
        return 0.0;
    }
    
    try {
        cv::Mat diff(seq1.size(), 1, CV_64F);
        for (size_t i = 0; i < seq1.size(); i++) {
            diff.at<double>(i) = seq1[i] - seq2[i];
        }
        
        // Add regularization to covariance matrix
        cv::Mat regularizedCov = covarianceMatrix_.clone();
        double epsilon = 1e-6 * cv::trace(covarianceMatrix_)[0];
        regularizedCov += cv::Mat::eye(regularizedCov.rows, regularizedCov.cols, CV_64F) * epsilon;
        
        cv::Mat invCovariance;
        if (!cv::invert(regularizedCov, invCovariance, cv::DECOMP_SVD)) {
            // If inversion still fails, fall back to weighted Euclidean distance
            double dist = 0.0;
            for (size_t i = 0; i < seq1.size(); i++) {
                double d = seq1[i] - seq2[i];
                dist += d * d;
            }
            return std::sqrt(dist);
        }
        
        cv::Mat distance = diff.t() * invCovariance * diff;
        return std::sqrt(std::abs(distance.at<double>(0, 0)));
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in Mahalanobis distance: " << e.what() << std::endl;
        return 0.0;
    }
}

double GaitClassifier::computeDTWDistance(
    const std::vector<double>& seq1,
    const std::vector<double>& seq2) const {
    
    size_t n = seq1.size();
    size_t m = seq2.size();
    
    std::vector<std::vector<double>> dtw(n + 1, 
        std::vector<double>(m + 1, std::numeric_limits<double>::infinity()));
    
    dtw[0][0] = 0.0;
    
    // Compute DTW matrix with Sakoe-Chiba band constraint
    int bandWidth = std::max<int>(static_cast<int>(std::abs(static_cast<int>(n) - static_cast<int>(m))) / 2, 5);
    
    for (size_t i = 1; i <= n; i++) {
        // Fix the max/min calls to handle size_t properly
        size_t j_start = i > static_cast<size_t>(bandWidth) ? i - bandWidth : 1;
        size_t j_end = std::min<size_t>(m, i + bandWidth);
        
        for (size_t j = j_start; j <= j_end; j++) {
            double cost = std::abs(seq1[i-1] - seq2[j-1]);
            dtw[i][j] = cost + std::min({
                dtw[i-1][j],      // insertion
                dtw[i][j-1],      // deletion
                dtw[i-1][j-1]     // match
            });
        }
    }
    
    return dtw[n][m] / static_cast<double>(std::max(n, m));  // Normalize by sequence length
}

void GaitClassifier::computeFeatureStatistics() {
    if (trainingData_.empty()) return;
    
    size_t numFeatures = trainingData_[0].size();
    featureMeans_.resize(numFeatures);
    featureStdDevs_.resize(numFeatures);
    
    // Calculate robust statistics for each feature
    for (size_t i = 0; i < numFeatures; i++) {
        std::vector<double> featureValues;
        featureValues.reserve(trainingData_.size());
        
        for (const auto& sample : trainingData_) {
            featureValues.push_back(sample[i]);
        }
        
        featureMeans_[i] = computeRobustMean(featureValues);
        featureStdDevs_[i] = computeRobustStd(featureValues, featureMeans_[i]);
    }
}

double GaitClassifier::computeRobustMean(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    
    std::vector<double> sortedValues = values;
    std::sort(sortedValues.begin(), sortedValues.end());
    
    // Use trimmed mean (remove top and bottom 10%)
    size_t trimCount = sortedValues.size() / 10;
    double sum = 0.0;
    size_t count = 0;
    
    for (size_t i = trimCount; i < sortedValues.size() - trimCount; i++) {
        sum += sortedValues[i];
        count++;
    }
    
    return count > 0 ? sum / count : 0.0;
}

double GaitClassifier::computeRobustStd(
    const std::vector<double>& values, double mean) const {
    
    if (values.empty()) return 1.0;
    
    std::vector<double> deviations;
    deviations.reserve(values.size());
    
    for (double val : values) {
        deviations.push_back(std::abs(val - mean));
    }
    
    std::sort(deviations.begin(), deviations.end());
    
    // Use median absolute deviation (MAD)
    double mad = deviations[deviations.size() / 2];
    
    // Scale MAD to approximate standard deviation
    return 1.4826 * mad;
}

void GaitClassifier::computeCovarianceMatrix() {
    if (trainingData_.empty() || trainingData_[0].empty()) {
        std::cerr << "Empty training data, cannot compute covariance matrix" << std::endl;
        return;
    }
    
    try {
        size_t numSamples = trainingData_.size();
        size_t numFeatures = trainingData_[0].size();
        
        // Create data matrix
        cv::Mat data = cv::Mat::zeros(numSamples, numFeatures, CV_64F);
        
        // Fill and normalize the data
        std::vector<double> featureMeans(numFeatures, 0.0);
        std::vector<double> featureStds(numFeatures, 0.0);
        
        // Compute means
        for (size_t j = 0; j < numFeatures; j++) {
            for (size_t i = 0; i < numSamples; i++) {
                featureMeans[j] += trainingData_[i][j];
            }
            featureMeans[j] /= numSamples;
        }
        
        // Compute standard deviations
        for (size_t j = 0; j < numFeatures; j++) {
            for (size_t i = 0; i < numSamples; i++) {
                double diff = trainingData_[i][j] - featureMeans[j];
                featureStds[j] += diff * diff;
            }
            featureStds[j] = std::sqrt(featureStds[j] / (numSamples - 1));
            if (featureStds[j] < 1e-10) featureStds[j] = 1.0;  // Prevent division by zero
        }
        
        // Fill normalized data matrix
        for (size_t i = 0; i < numSamples; i++) {
            for (size_t j = 0; j < numFeatures; j++) {
                data.at<double>(i, j) = (trainingData_[i][j] - featureMeans[j]) / featureStds[j];
            }
        }
        
        // Compute robust covariance
        covarianceMatrix_ = (data.t() * data) / (numSamples - 1);
        
        // Add small regularization term
        double trace = cv::trace(covarianceMatrix_)[0];
        double epsilon = 1e-6 * trace / numFeatures;
        covarianceMatrix_ += cv::Mat::eye(numFeatures, numFeatures, CV_64F) * epsilon;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error in covariance computation: " << e.what() << std::endl;
        covarianceMatrix_ = cv::Mat();
    }
}

std::vector<double> GaitClassifier::normalizeFeatures(const std::vector<double>& features) const {
    if (features.empty()) return features;
    
    // Calculate mean and standard deviation
    double sum = 0.0, sqSum = 0.0;
    for (double f : features) {
        sum += f;
        sqSum += f * f;
    }
    
    double mean = sum / features.size();
    double variance = (sqSum / features.size()) - (mean * mean);
    double stdDev = std::sqrt(std::max(variance, 1e-10));
    
    // Z-score normalization
    std::vector<double> normalized(features.size());
    for (size_t i = 0; i < features.size(); i++) {
        normalized[i] = (features[i] - mean) / stdDev;
    }
    
    return normalized;
}

void GaitClassifier::saveModel(const std::string& filename) const {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Save parameters
    fs << "min_confidence" << params_.minConfidenceThreshold;
    fs << "k_neighbors" << params_.kNeighbors;
    fs << "max_distance" << params_.maxValidDistance;
    fs << "temporal_weight" << params_.temporalWeight;
    fs << "spatial_weight" << params_.spatialWeight;
    
    // Save feature statistics
    fs << "feature_means" << featureMeans_;
    fs << "feature_stddevs" << featureStdDevs_;
    fs << "covariance_matrix" << covarianceMatrix_;
    
    // Save training data
    fs << "num_samples" << (int)trainingData_.size();
    fs << "feature_size" << (int)featureMeans_.size();
    
    // Save training data and labels
    for (size_t i = 0; i < trainingData_.size(); i++) {
        fs << "sample_" + std::to_string(i) << trainingData_[i];
        fs << "label_" + std::to_string(i) << trainingLabels_[i];
    }
    
    fs.release();
}

bool GaitClassifier::loadModel(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "Failed to open model file: " << filename << std::endl;
        return false;
    }
    
    try {
        // Load parameters
        fs["min_confidence"] >> params_.minConfidenceThreshold;
        fs["k_neighbors"] >> params_.kNeighbors;
        fs["max_distance"] >> params_.maxValidDistance;
        fs["temporal_weight"] >> params_.temporalWeight;
        fs["spatial_weight"] >> params_.spatialWeight;
        
        // Load feature statistics
        fs["feature_means"] >> featureMeans_;
        fs["feature_stddevs"] >> featureStdDevs_;
        fs["covariance_matrix"] >> covarianceMatrix_;
        
        // Load training data
        int numSamples, featureSize;
        fs["num_samples"] >> numSamples;
        fs["feature_size"] >> featureSize;
        
        trainingData_.clear();
        trainingLabels_.clear();
        
        // Load samples and labels
        for (int i = 0; i < numSamples; i++) {
            std::vector<double> sample;
            std::string label;
            
            fs["sample_" + std::to_string(i)] >> sample;
            fs["label_" + std::to_string(i)] >> label;
            
            trainingData_.push_back(sample);
            trainingLabels_.push_back(label);
        }
        
        isModelTrained_ = true;
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

double GaitClassifier::computeEuclideanDistance(
    const std::vector<double>& seq1,
    const std::vector<double>& seq2) const {
    
    if (seq1.empty() || seq2.empty() || seq1.size() != seq2.size()) {
        return std::numeric_limits<double>::max();
    }

    double dist = 0.0;
    for (size_t i = 0; i < seq1.size(); i++) {
        double diff = seq1[i] - seq2[i];
        dist += diff * diff;
    }
    
    return std::sqrt(dist);
}

double computeRobustConfidence(
    const std::vector<std::pair<double, std::string>>& distances,
    const std::string& bestMatch) {
    
    if (distances.empty()) return 0.0;
    
    // Get intra-class and inter-class distances
    std::vector<double> intraClassDists;
    std::vector<double> interClassDists;
    
    for (const auto& [dist, label] : distances) {
        if (label == bestMatch) {
            intraClassDists.push_back(dist);
        } else {
            interClassDists.push_back(dist);
        }
    }
    
    if (intraClassDists.empty() || interClassDists.empty()) {
        return 0.5;  // Uncertain case
    }
    
    // Compute statistics
    double meanIntra = std::accumulate(intraClassDists.begin(), 
                                        intraClassDists.end(), 0.0) / intraClassDists.size();
    double meanInter = std::accumulate(interClassDists.begin(), 
                                        interClassDists.end(), 0.0) / interClassDists.size();
    
    // Compute confidence based on separation between classes
    double separation = (meanInter - meanIntra) / (meanInter + meanIntra);
    
    // Map to [0.5, 1.0] range with sigmoid-like function
    return 0.5 + 0.5 * (1.0 - std::exp(-5.0 * separation)) / 
                        (1.0 + std::exp(-5.0 * separation));
}


} // namespace gait
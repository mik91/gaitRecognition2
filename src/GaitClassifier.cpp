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
    
    if (personFeatures.empty()) {
        std::cerr << "No person features provided for training" << std::endl;
        return false;
    }
    
    try {
        // Clear existing data
        trainingSequences_.clear();
        trainingData_.clear();
        trainingLabels_.clear();
        
        std::cout << "\nStarting training process..." << std::endl;
        std::cout << "Number of subjects: " << personFeatures.size() << std::endl;
        
        // Process each person's data
        for (const auto& [person, sequences] : personFeatures) {
            std::cout << "Processing subject " << person << ": " 
                      << sequences.size() << " sequences" << std::endl;
                      
            for (const auto& [features, filename] : sequences) {
                if (!features.empty()) {
                    // Store sequence info
                    SequenceInfo info;
                    info.label = person;
                    info.condition = extractCondition(filename);
                    info.features = features;
                    trainingSequences_.push_back(info);
                    
                    // Also update legacy data structures
                    trainingData_.push_back(features);
                    trainingLabels_.push_back(person);
                }
            }
        }
        
        if (trainingSequences_.empty()) {
            std::cerr << "No valid sequences after processing" << std::endl;
            return false;
        }
        
        std::cout << "Computing feature statistics..." << std::endl;
        computeFeatureStatistics();
        
        std::cout << "Computing covariance matrix..." << std::endl;
        computeCovarianceMatrix();
        
        isModelTrained_ = true;
        std::cout << "Training completed successfully" << std::endl;
        std::cout << "Total training sequences: " << trainingSequences_.size() << std::endl;
        std::cout << "Feature dimension: " << featureMeans_.size() << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        isModelTrained_ = false;
        return false;
    }
}

std::pair<std::string, double> GaitClassifier::identifyPerson(
    const std::vector<double>& testSequence,
    const std::string& testFilename) {
    
    if (!isModelTrained_) {
        std::cerr << "Model not trained! Please check if model file exists and is loaded properly" << std::endl;
        return {"unknown", 0.0};
    }

    // Validate input sequence
    if (testSequence.empty()) {
        std::cerr << "Empty test sequence provided" << std::endl;
        return {"unknown", 0.0};
    }

    // Validate feature dimensions
    if (testSequence.size() != featureMeans_.size()) {
        std::cerr << "Feature size mismatch! Expected " << featureMeans_.size() 
                  << " but got " << testSequence.size() << std::endl;
        return {"unknown", 0.0};
    }

    // Validate training data
    if (trainingSequences_.empty()) {
        std::cerr << "No training sequences available" << std::endl;
        return {"unknown", 0.0};
    }

    // Set maximum computation time
    const auto startTime = std::chrono::steady_clock::now();
    const auto timeoutDuration = std::chrono::seconds(30);

    try {
        std::string testCondition = extractCondition(testFilename);
        auto normalizedTest = normalizeFeatures(testSequence);
        
        // Pre-allocate vector with proper size
        std::vector<std::pair<double, SequenceInfo>> allDistances;
        allDistances.reserve(trainingSequences_.size());

        // Calculate distances
        for (const auto& trainSeq : trainingSequences_) {
            // Check timeout
            if (std::chrono::steady_clock::now() - startTime > timeoutDuration) {
                std::cerr << "Classification timed out after 30 seconds" << std::endl;
                return {"unknown", 0.0};
            }

            auto normalizedTrain = normalizeFeatures(trainSeq.features);
            double distance = computeEuclideanDistance(normalizedTest, normalizedTrain);
            
            // Only add valid distances
            if (!std::isnan(distance) && !std::isinf(distance)) {
                allDistances.emplace_back(distance, trainSeq);
            }
        }

        // Validate we have enough distances
        if (allDistances.empty()) {
            std::cerr << "No valid distances computed" << std::endl;
            return {"unknown", 0.0};
        }

        if (allDistances.size() < params_.kNeighbors) {
            std::cerr << "Not enough valid sequences for k-NN comparison" << std::endl;
            return {"unknown", 0.0};
        }

        // Sort distances
        std::sort(allDistances.begin(), allDistances.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });

        // Get k nearest neighbors
        std::map<std::string, int> votes;
        std::string bestMatch = "unknown";
        int maxVotes = 0;
        
        for (size_t i = 0; i < params_.kNeighbors && i < allDistances.size(); i++) {
            const auto& [dist, info] = allDistances[i];
            votes[info.label]++;
            
            if (votes[info.label] > maxVotes) {
                maxVotes = votes[info.label];
                bestMatch = info.label;
            }
        }

        // Compute confidence
        double confidence = computeConditionAwareConfidence(
            testCondition, 
            std::vector<std::pair<double, SequenceInfo>>(
                allDistances.begin(),
                allDistances.begin() + std::min(allDistances.size(), 
                                              static_cast<size_t>(params_.kNeighbors))),
            bestMatch);

        return {bestMatch, confidence};

    } catch (const std::exception& e) {
        std::cerr << "Error during classification: " << e.what() << std::endl;
        return {"unknown", 0.0};
    }
}

double GaitClassifier::computeConditionAwareConfidence(
    const std::string& testCondition,
    const std::vector<std::pair<double, SequenceInfo>>& distances,
    const std::string& predictedClass) const {
    
    if (distances.empty() || distances.size() < params_.kNeighbors) {
        return 0.0;
    }
    
    // Analyze k-nearest neighbors
    size_t k = std::min(static_cast<size_t>(params_.kNeighbors), distances.size());
    int matchCount = 0;
    double totalMatchDistance = 0.0;
    double totalDistance = 0.0;
    std::set<std::string> uniqueClasses;
    
    // First pass: collect statistics
    for (size_t i = 0; i < k; i++) {
        const auto& [dist, info] = distances[i];
        uniqueClasses.insert(info.label);
        totalDistance += dist;
        
        if (info.label == predictedClass) {
            matchCount++;
            totalMatchDistance += dist;
        }
    }
    
    // Early rejection for scattered predictions
    if (uniqueClasses.size() > 3 || matchCount == 0) {
        return 0.0;
    }
    
    // Calculate base metrics
    double matchRatio = static_cast<double>(matchCount) / k;
    double avgMatchDistance = matchCount > 0 ? totalMatchDistance / matchCount : 0.0;
    double avgDistance = totalDistance / k;
    
    // Check if distances are too large (unknown subject)
    if (avgDistance > params_.maxValidDistance * 0.7) {
        return 0.0;
    }
    
    // Calculate confidence components
    double matchConfidence = 1.0 / (1.0 + std::exp(-5.0 * (matchRatio - 0.5)));
    double distanceConfidence = std::exp(-avgMatchDistance / params_.maxValidDistance);
    double distributionConfidence = 1.0 - (static_cast<double>(uniqueClasses.size() - 1) / k);
    
    // Weighted combination of confidence components
    double confidence = 0.4 * matchConfidence + 
                       0.4 * distanceConfidence + 
                       0.2 * distributionConfidence;
    
    // Adjust for condition match
    if (testCondition == distances[0].second.condition) {
        confidence *= 1.05;  // Reduced bonus from 1.1 to 1.05
    }
    
    // Additional penalties
    if (matchRatio < 0.5) {
        confidence *= 0.8;  // Penalty for minority predictions
    }
    
    if (uniqueClasses.size() > 1) {
        confidence *= (1.0 - 0.1 * (uniqueClasses.size() - 1));  // Penalty for class diversity
    }
    
    // Scale to reasonable range and cap maximum
    confidence = 0.6 + 0.35 * confidence;  // Scale to [0.6, 0.95] range
    confidence = std::min(0.95, confidence);  // Cap maximum at 0.95
    
    // Final threshold for unknown subjects
    if (confidence < params_.minConfidenceThreshold) {
        return 0.0;
    }
    
    return confidence;
}
// 0.7 * baseConfidence + 0.3 * distanceConfidence;
    // Calculate base confidence from match ratio
    // double matchRatio = static_cast<double>(matchCount) / k;
    // double avgMatchDistance = totalMatchDistance / matchCount;
    
    // // Calculate confidence using sigmoid function
    // double baseConfidence = 1.0 / (1.0 + std::exp(-5.0 * (matchRatio - 0.5)));
    
    // // Adjust confidence based on distance
    // double distanceConfidence = std::exp(-avgMatchDistance / params_.maxValidDistance);
    
    // // Final confidence calculation
    // double confidence = 0.7 * baseConfidence + 0.3 * distanceConfidence;
    
    // // Add small bonus for condition match
    // if (testCondition == distances[0].second.condition) {
    //     confidence *= 1.1;
    // }
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
    
    // Save training sequences info
    fs << "num_sequences" << (int)trainingSequences_.size();
    fs << "feature_size" << (int)featureMeans_.size();
    
    // Save training sequences
    for (size_t i = 0; i < trainingSequences_.size(); i++) {
        const auto& seq = trainingSequences_[i];
        std::string prefix = "sequence_" + std::to_string(i) + "_";
        fs << prefix + "label" << seq.label;
        fs << prefix + "condition" << seq.condition;
        fs << prefix + "features" << seq.features;
    }
    
    // For backwards compatibility, also save training data and labels
    fs << "num_samples" << (int)trainingData_.size();
    for (size_t i = 0; i < trainingData_.size(); i++) {
        fs << "sample_" + std::to_string(i) << trainingData_[i];
        fs << "label_" + std::to_string(i) << trainingLabels_[i];
    }
    
    fs.release();
}

bool GaitClassifier::loadModel(const std::string& filename) {
    std::cout << "Loading model from: " << filename << std::endl;
    
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open model file: " << filename << std::endl;
        return false;
    }
    
    try {
        // Clear existing data
        trainingSequences_.clear();
        trainingData_.clear();
        trainingLabels_.clear();
        
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
        
        // Load training sequences
        int numSequences = 0;
        fs["num_sequences"] >> numSequences;
        
        if (numSequences > 0) {
            trainingSequences_.reserve(numSequences);
            
            for (int i = 0; i < numSequences; i++) {
                std::string prefix = "sequence_" + std::to_string(i) + "_";
                SequenceInfo seq;
                
                fs[prefix + "label"] >> seq.label;
                fs[prefix + "condition"] >> seq.condition;
                fs[prefix + "features"] >> seq.features;
                
                if (!seq.label.empty() && !seq.features.empty()) {
                    trainingSequences_.push_back(std::move(seq));
                }
            }
        }
        
        // Load legacy training data
        int numSamples = 0;
        fs["num_samples"] >> numSamples;
        
        if (numSamples > 0) {
            trainingData_.reserve(numSamples);
            trainingLabels_.reserve(numSamples);
            
            for (int i = 0; i < numSamples; i++) {
                std::vector<double> sample;
                std::string label;
                
                fs["sample_" + std::to_string(i)] >> sample;
                fs["label_" + std::to_string(i)] >> label;
                
                if (!sample.empty() && !label.empty()) {
                    trainingData_.push_back(std::move(sample));
                    trainingLabels_.push_back(std::move(label));
                }
            }
        }
        
        // Validate loaded data
        if (trainingSequences_.empty()) {
            std::cerr << "Warning: No training sequences loaded" << std::endl;
            return false;
        }
        
        isModelTrained_ = true;
        std::cout << "Model loaded successfully" << std::endl;
        std::cout << "Loaded " << trainingSequences_.size() << " sequences" << std::endl;
        std::cout << "Feature dimension: " << featureMeans_.size() << std::endl;
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error loading model: " << e.what() << std::endl;
        isModelTrained_ = false;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        isModelTrained_ = false;
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
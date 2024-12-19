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
        trainingSequences_.clear();
        trainingData_.clear();
        trainingLabels_.clear();
        
        std::cout << "\nStarting training process..." << std::endl;
        std::cout << "Number of subjects: " << personFeatures.size() << std::endl;
        
        for (const auto& [person, sequences] : personFeatures) {
            std::cout << "Processing subject " << person << ": " 
                      << sequences.size() << " sequences" << std::endl;
                      
            for (const auto& [features, filename] : sequences) {
                if (!features.empty()) {
                    SequenceInfo info;
                    info.label = person;
                    info.condition = extractCondition(filename);
                    info.features = features;
                    trainingSequences_.push_back(info);
                    
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

    if (testSequence.empty()) {
        std::cerr << "Empty test sequence provided" << std::endl;
        return {"unknown", 0.0};
    }

    if (testSequence.size() != featureMeans_.size()) {
        std::cerr << "Feature size mismatch! Expected " << featureMeans_.size() 
                  << " but got " << testSequence.size() << std::endl;
        return {"unknown", 0.0};
    }

    if (trainingSequences_.empty()) {
        std::cerr << "No training sequences available" << std::endl;
        return {"unknown", 0.0};
    }

    // Maximum computation time
    const auto startTime = std::chrono::steady_clock::now();
    const auto timeoutDuration = std::chrono::seconds(30);

    try {
        std::string testCondition = extractCondition(testFilename);
        auto normalizedTest = normalizeFeatures(testSequence);
        
        std::vector<std::pair<double, SequenceInfo>> allDistances;
        allDistances.reserve(trainingSequences_.size());

        // Calculate distances
        for (const auto& trainSeq : trainingSequences_) {
            if (std::chrono::steady_clock::now() - startTime > timeoutDuration) {
                std::cerr << "Classification timed out after 30 seconds" << std::endl;
                return {"unknown", 0.0};
            }

            auto normalizedTrain = normalizeFeatures(trainSeq.features);
            double distance = computeEuclideanDistance(normalizedTest, normalizedTrain);
            
            if (!std::isnan(distance) && !std::isinf(distance)) {
                allDistances.emplace_back(distance, trainSeq);
            }
        }

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
    
    for (size_t i = 0; i < k; i++) {
        const auto& [dist, info] = distances[i];
        uniqueClasses.insert(info.label);
        totalDistance += dist;
        
        if (info.label == predictedClass) {
            matchCount++;
            totalMatchDistance += dist;
        }
    }
    
    if (uniqueClasses.size() > 3 || matchCount == 0) {
        return 0.0;
    }
    
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
    
    double confidence = 0.4 * matchConfidence + 
                       0.4 * distanceConfidence + 
                       0.2 * distributionConfidence;
    
    if (testCondition == distances[0].second.condition) {
        confidence *= 1.05;
    }
    
    if (matchRatio < 0.5) {
        // Penalty for minority predictions
        confidence *= 0.8;  
    }
    
    if (uniqueClasses.size() > 1) {
        // Penalty for class diversity
        confidence *= (1.0 - 0.1 * (uniqueClasses.size() - 1));  
    }
    
    confidence = 0.6 + 0.35 * confidence;
    confidence = std::min(0.95, confidence);
    
    if (confidence < params_.minConfidenceThreshold) {
        return 0.0;
    }
    
    return confidence;
}

void GaitClassifier::computeFeatureStatistics() {
    if (trainingData_.empty()) return;
    
    size_t numFeatures = trainingData_[0].size();
    featureMeans_.resize(numFeatures);
    featureStdDevs_.resize(numFeatures);
    
    for (size_t i = 0; i < numFeatures; i++) {
        std::vector<double> featureValues;
        featureValues.reserve(trainingData_.size());
        
        for (const auto& sample : trainingData_) {
            featureValues.push_back(sample[i]);
        }
        
        featureMeans_[i] = computeMean(featureValues);
        featureStdDevs_[i] = computeStd(featureValues, featureMeans_[i]);
    }
}

double GaitClassifier::computeMean(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    
    std::vector<double> sortedValues = values;
    std::sort(sortedValues.begin(), sortedValues.end());
    
    size_t trimCount = sortedValues.size() / 10;
    double sum = 0.0;
    size_t count = 0;
    
    for (size_t i = trimCount; i < sortedValues.size() - trimCount; i++) {
        sum += sortedValues[i];
        count++;
    }
    
    return count > 0 ? sum / count : 0.0;
}

/*
* Compute the median absolute deviation (MAD) of a set of values
*/
double GaitClassifier::computeStd(
    const std::vector<double>& values, double mean) const {
    
    if (values.empty()) return 1.0;
    
    std::vector<double> deviations;
    deviations.reserve(values.size());
    
    for (double val : values) {
        deviations.push_back(std::abs(val - mean));
    }
    
    std::sort(deviations.begin(), deviations.end());
    
    double mad = deviations[deviations.size() / 2];
    
    return 1.4826 * mad;
}

/*
* Compute the covariance matrix of the training data
*/
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
        
        std::vector<double> featureMeans(numFeatures, 0.0);
        std::vector<double> featureStds(numFeatures, 0.0);
        
        for (size_t j = 0; j < numFeatures; j++) {
            for (size_t i = 0; i < numSamples; i++) {
                featureMeans[j] += trainingData_[i][j];
            }
            featureMeans[j] /= numSamples;
        }
        
        for (size_t j = 0; j < numFeatures; j++) {
            for (size_t i = 0; i < numSamples; i++) {
                double diff = trainingData_[i][j] - featureMeans[j];
                featureStds[j] += diff * diff;
            }
            featureStds[j] = std::sqrt(featureStds[j] / (numSamples - 1));
            if (featureStds[j] < 1e-10) featureStds[j] = 1.0;
        }
        
        for (size_t i = 0; i < numSamples; i++) {
            for (size_t j = 0; j < numFeatures; j++) {
                data.at<double>(i, j) = (trainingData_[i][j] - featureMeans[j]) / featureStds[j];
            }
        }
        
        covarianceMatrix_ = (data.t() * data) / (numSamples - 1);
        
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
    
    for (size_t i = 0; i < trainingSequences_.size(); i++) {
        const auto& seq = trainingSequences_[i];
        std::string prefix = "sequence_" + std::to_string(i) + "_";
        fs << prefix + "label" << seq.label;
        fs << prefix + "condition" << seq.condition;
        fs << prefix + "features" << seq.features;
    }
    
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

} // namespace gait
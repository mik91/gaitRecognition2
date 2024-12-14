// GaitClassifier.h
#pragma once
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include <memory>

namespace gait {

struct ClassifierParams {
    double minConfidenceThreshold;   
    double sequenceConfidenceThreshold; 
    int kNeighbors;                     
    double maxValidDistance;        
    double temporalWeight;            
    double spatialWeight;           
    
    // ClassifierParams() = default;
    ClassifierParams(double minConf = 0.65, 
                    int k = 5, 
                    double maxDist = 100.0,
                    double tempW = 0.5, 
                    double spatW = 0.5)
        : minConfidenceThreshold(minConf)
        , sequenceConfidenceThreshold(0.60)  // Default to 0.60 for sequences
        , kNeighbors(k)
        , maxValidDistance(maxDist)
        , temporalWeight(tempW)
        , spatialWeight(spatW) {}
    // ClassifierParams(double minConf, int k, double maxDist, 
    //                 double tempW, double spatW)
    //     : minConfidenceThreshold(minConf), kNeighbors(k), maxValidDistance(maxDist),
    //       temporalWeight(tempW), spatialWeight(spatW) {}
};

class GaitClassifier {
public:
    explicit GaitClassifier(const ClassifierParams& params = ClassifierParams());

    // Training methods
    bool analyzePatterns(
        const std::map<std::string, 
                      std::vector<std::pair<std::vector<double>, std::string>>>& personFeatures);

    // Recognition methods
    std::pair<std::string, double> identifyPerson(
        const std::vector<double>& testSequence,
        const std::string& testFilename);

    // Model state
    bool isModelTrained() const { return isModelTrained_; }
    
    // Utility methods
    std::vector<double> normalizeFeatures(const std::vector<double>& features) const;
    size_t getNumFeatures() const { return featureMeans_.size(); }
    void saveModel(const std::string& filename) const;
    bool loadModel(const std::string& filename);

private:
    struct SequenceInfo {
        std::string label;
        std::string condition;
        std::vector<double> features;

        // Add comparison operators
        bool operator<(const SequenceInfo& other) const {
            if (label != other.label) return label < other.label;
            if (condition != other.condition) return condition < other.condition;
            return features < other.features;
        }

        bool operator==(const SequenceInfo& other) const {
            return label == other.label && 
                   condition == other.condition && 
                   features == other.features;
        }
    };
    // Core methods
    std::string extractCondition(const std::string& filename) const;
    double computeEuclideanDistance(const std::vector<double>& seq1, 
                                  const std::vector<double>& seq2) const;
    double computeDTWDistance(const std::vector<double>& seq1,
                            const std::vector<double>& seq2) const;
    double computeMahalanobisDistance(const std::vector<double>& seq1,
                                    const std::vector<double>& seq2) const;
    
    // Confidence computation
    double computeConditionAwareConfidence(
        const std::string& testCondition,
        const std::vector<std::pair<double, SequenceInfo>>& distances,
        const std::string& predictedClass) const;

    double computeRobustConfidence(
        const std::vector<std::pair<double, std::string>>& distances,
        const std::string& bestMatch) const;

    // Feature processing
    void computeFeatureStatistics();
    double computeRobustMean(const std::vector<double>& values) const;
    double computeRobustStd(const std::vector<double>& values, double mean) const;
    void computeCovarianceMatrix();

    // Model data
    std::vector<SequenceInfo> trainingSequences_;
    std::vector<std::vector<double>> trainingData_;
    std::vector<std::string> trainingLabels_;
    std::vector<double> featureMeans_;
    std::vector<double> featureStdDevs_;
    cv::Mat covarianceMatrix_;
    ClassifierParams params_;
    bool isModelTrained_;
};

} // namespace gait
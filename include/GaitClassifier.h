#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include "GaitAnalyzer.h"

namespace gait {

class GaitClassifier {
public:
    // Enum to represent different gait types
    enum class GaitType {
        NORMAL,
        ABNORMAL
    };

    // Constructor
    GaitClassifier();

    // Training method
    bool train(const std::vector<std::vector<double>>& features, 
               const std::vector<GaitType>& labels);

    // Classification methods
    GaitType classify(const std::vector<double>& features);
    double getConfidence(const std::vector<double>& features);

    // Model persistence
    bool saveModel(const std::string& filename);
    bool loadModel(const std::string& filename);

private:
    cv::Ptr<cv::ml::SVM> svm_;
    bool isModelTrained_;
};

class GaitAnalysisSystem {
public:
    explicit GaitAnalysisSystem(const SymmetryParams& params);

    // Add getter for the analyzer
    GaitAnalyzer& getAnalyzer() { return analyzer_; }
    const GaitAnalyzer& getAnalyzer() const { return analyzer_; }

    GaitClassifier::GaitType analyzeSequence(
        const std::vector<cv::Mat>& frames,
        double& confidence);

    bool trainClassifier(
        const std::vector<std::vector<cv::Mat>>& normalSequences,
        const std::vector<std::vector<cv::Mat>>& abnormalSequences);

    bool saveClassifier(const std::string& filename);
    bool loadClassifier(const std::string& filename);

private:
    GaitAnalyzer analyzer_;
    GaitClassifier classifier_;

    void processSequencesForTraining(
        const std::vector<std::vector<cv::Mat>>& sequences,
        GaitClassifier::GaitType label,
        std::vector<std::vector<double>>& allFeatures,
        std::vector<GaitClassifier::GaitType>& labels);
};

// Optional: Utility functions for cross-validation and evaluation
struct EvaluationMetrics {
    double accuracy;
    double precision;
    double recall;
    double f1Score;
    double falsePositiveRate;
};

class GaitEvaluator {
public:
    static EvaluationMetrics crossValidate(
        GaitAnalysisSystem& system,
        const std::vector<std::vector<cv::Mat>>& normalSequences,
        const std::vector<std::vector<cv::Mat>>& abnormalSequences,
        int folds = 5);

    static EvaluationMetrics evaluate(
        GaitAnalysisSystem& system,
        const std::vector<std::vector<cv::Mat>>& testSequences,
        const std::vector<GaitClassifier::GaitType>& trueLabels);

private:
    static std::vector<std::vector<size_t>> createFolds(
        size_t datasetSize, 
        int numFolds);
};

} // namespace gait

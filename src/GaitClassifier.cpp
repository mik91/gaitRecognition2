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

    GaitClassifier() : isModelTrained_(false) {
        // Initialize SVM with optimal parameters for gait classification
        svm_ = cv::ml::SVM::create();
        svm_->setType(cv::ml::SVM::C_SVC);
        svm_->setKernel(cv::ml::SVM::RBF);
        svm_->setTermCriteria(cv::TermCriteria(
            cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
            1000, 1e-6));
    }

    // Train the classifier with labeled data
    bool train(const std::vector<std::vector<double>>& features, 
               const std::vector<GaitType>& labels) {
        if (features.empty() || features.size() != labels.size()) {
            return false;
        }

        // Convert features and labels to OpenCV matrices
        cv::Mat featuresMat(features.size(), features[0].size(), CV_32F);
        cv::Mat labelsMat(labels.size(), 1, CV_32S);

        // Fill the matrices
        for (size_t i = 0; i < features.size(); i++) {
            for (size_t j = 0; j < features[i].size(); j++) {
                featuresMat.at<float>(i, j) = static_cast<float>(features[i][j]);
            }
            labelsMat.at<int>(i) = (labels[i] == GaitType::NORMAL) ? 1 : -1;
        }

        // Train the SVM
        try {
            svm_->train(featuresMat, cv::ml::ROW_SAMPLE, labelsMat);
            isModelTrained_ = true;
            return true;
        } catch (const cv::Exception& e) {
            std::cerr << "Training error: " << e.what() << std::endl;
            return false;
        }
    }

    // Classify a single sequence of gait features
    GaitType classify(const std::vector<double>& features) {
        if (!isModelTrained_ || features.empty()) {
            throw std::runtime_error("Model not trained or invalid features");
        }

        // Convert features to OpenCV matrix
        cv::Mat featureMat(1, features.size(), CV_32F);
        for (size_t i = 0; i < features.size(); i++) {
            featureMat.at<float>(0, i) = static_cast<float>(features[i]);
        }

        // Predict using SVM
        float response = svm_->predict(featureMat);
        return (response > 0) ? GaitType::NORMAL : GaitType::ABNORMAL;
    }

    // Get confidence score for classification
    double getConfidence(const std::vector<double>& features) {
        if (!isModelTrained_ || features.empty()) {
            return 0.0;
        }

        cv::Mat featureMat(1, features.size(), CV_32F);
        for (size_t i = 0; i < features.size(); i++) {
            featureMat.at<float>(0, i) = static_cast<float>(features[i]);
        }

        // Get confidence score using decision function
        cv::Mat results;
        svm_->predict(featureMat, results, cv::ml::StatModel::RAW_OUTPUT);
        return std::abs(results.at<float>(0, 0));
    }

    // Save trained model
    bool saveModel(const std::string& filename) {
        if (!isModelTrained_) {
            return false;
        }
        try {
            svm_->save(filename);
            return true;
        } catch (const cv::Exception& e) {
            std::cerr << "Error saving model: " << e.what() << std::endl;
            return false;
        }
    }

    // Load trained model
    bool loadModel(const std::string& filename) {
        try {
            svm_ = cv::ml::SVM::load(filename);
            isModelTrained_ = true;
            return true;
        } catch (const cv::Exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    }

private:
    cv::Ptr<cv::ml::SVM> svm_;
    bool isModelTrained_;
};

// Utility class to combine analysis and classification
class GaitAnalysisSystem {
public:
    GaitAnalysisSystem(const SymmetryParams& params) 
        : analyzer_(params) {}

    // Process a sequence of frames and classify the gait
    GaitClassifier::GaitType analyzeSequence(
        const std::vector<cv::Mat>& frames,
        double& confidence) {
        
        std::vector<double> aggregatedFeatures;
        
        // Process each frame and accumulate features
        for (const auto& frame : frames) {
            cv::Mat symmetryMap = analyzer_.processFrame(frame);
            auto frameFeatures = analyzer_.extractGaitFeatures(symmetryMap);
            
            // Initialize or update aggregated features
            if (aggregatedFeatures.empty()) {
                aggregatedFeatures = frameFeatures;
            } else {
                for (size_t i = 0; i < frameFeatures.size(); i++) {
                    aggregatedFeatures[i] += frameFeatures[i];
                }
            }
        }

        // Average the features
        for (auto& feature : aggregatedFeatures) {
            feature /= frames.size();
        }

        // Get classification and confidence
        confidence = classifier_.getConfidence(aggregatedFeatures);
        return classifier_.classify(aggregatedFeatures);
    }

    // Train the classifier
    bool trainClassifier(
        const std::vector<std::vector<cv::Mat>>& normalSequences,
        const std::vector<std::vector<cv::Mat>>& abnormalSequences) {
        
        std::vector<std::vector<double>> allFeatures;
        std::vector<GaitClassifier::GaitType> labels;

        // Process normal sequences
        processSequencesForTraining(normalSequences, 
                                  GaitClassifier::GaitType::NORMAL,
                                  allFeatures, labels);

        // Process abnormal sequences
        processSequencesForTraining(abnormalSequences, 
                                  GaitClassifier::GaitType::ABNORMAL,
                                  allFeatures, labels);

        // Train the classifier
        return classifier_.train(allFeatures, labels);
    }

    bool saveClassifier(const std::string& filename) {
        return classifier_.saveModel(filename);
    }

    bool loadClassifier(const std::string& filename) {
        return classifier_.loadModel(filename);
    }

private:
    GaitAnalyzer analyzer_;
    GaitClassifier classifier_;

    void processSequencesForTraining(
        const std::vector<std::vector<cv::Mat>>& sequences,
        GaitClassifier::GaitType label,
        std::vector<std::vector<double>>& allFeatures,
        std::vector<GaitClassifier::GaitType>& labels) {
        
        for (const auto& sequence : sequences) {
            std::vector<double> aggregatedFeatures;
            
            for (const auto& frame : sequence) {
                cv::Mat symmetryMap = analyzer_.processFrame(frame);
                auto frameFeatures = analyzer_.extractGaitFeatures(symmetryMap);
                
                if (aggregatedFeatures.empty()) {
                    aggregatedFeatures = frameFeatures;
                } else {
                    for (size_t i = 0; i < frameFeatures.size(); i++) {
                        aggregatedFeatures[i] += frameFeatures[i];
                    }
                }
            }

            // Average the features
            for (auto& feature : aggregatedFeatures) {
                feature /= sequence.size();
            }

            allFeatures.push_back(aggregatedFeatures);
            labels.push_back(label);
        }
    }
};

} // namespace gait

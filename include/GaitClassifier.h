#pragma once
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

namespace gait {

class GaitClassifier {
public:
    // Structure to hold a person's gait sequence data
    struct PersonData {
        std::string id;
        std::vector<std::vector<double>> features;
    };

    GaitClassifier();
    
    // Training and classification
    bool analyzePatterns(const std::map<std::string, std::vector<std::vector<double>>>& personFeatures, 
                         bool visualize = false);
    std::pair<std::string, double> identifyPerson(const std::vector<double>& testSequence, 
                                                  bool visualize = false);
    
    // Visualization methods
    void visualizeTrainingData(const std::string& windowName = "Training Data");
    void visualizeClassification(const std::vector<double>& testSequence, 
                               const std::string& windowName = "Classification Result");
    
    // Getters
    bool isModelTrained() const { return isModelTrained_; }
    std::string getClusterStats() const;

private:
    double computeDistance(const std::vector<double>& seq1, const std::vector<double>& seq2);
    void computeClusterStatistics(const cv::Mat& features, const cv::Mat& labels);

    struct ClusterStats {
        int size = 0;
        double avgDistance = 0.0;
        double maxDistance = 0.0;
        double variance = 0.0;
    };

    std::vector<std::vector<double>> trainingData_;
    std::vector<std::string> trainingLabels_;
    std::vector<ClusterStats> clusterStats_;
    bool isModelTrained_;
};

} // namespace gait
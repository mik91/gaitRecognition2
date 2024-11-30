#include "Loader.h"
#include <iostream>

int main() {
    try {
        // Windows path format
        #ifdef _WIN32
            gait::Loader loader("C:\\Users\\kamar\\OneDrive\\Documents\\UDEM\\Session 1\\IFT6150\\gaitRecognition2\\data\\CASIA_B");
        #else
            // Linux/Unix path format
            gait::Loader loader("/u/kamarami/Documents/linux-gaitanalyzer/data/CASIA_B");
        #endif
        
        // Print available subjects
        std::cout << "Available subjects:" << std::endl;
        for (const auto& subject : loader.getSubjectIds()) {
            std::cout << " - " << subject << std::endl;
        }
        
        // Try to load a sequence
        try {
            auto frames = loader.loadSequence("test", "bg", 1);
            std::cout << "Successfully loaded " << frames.size() << " frames" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error loading sequence: " << e.what() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing loader: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
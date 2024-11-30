#include <filesystem>
namespace fs = std::filesystem;
#include <iostream>
#include <vector>
#include "Loader.h"


int main() {
    std::cout << "Loading images for subject test..." << std::endl;
    gait::Loader loader;

    auto sequences = loader.loadSequence("test", "bg", 1);
    return 0;
}

cmake_minimum_required(VERSION 3.10)
project(GaitClassification)

set(CMAKE_CXX_STANDARD 17)

# Locate OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include additional header files from the include directory
include_directories(include)

# Add source files
add_executable(
    ${PROJECT_NAME}
    src/GaitClassifier.cpp
    src/GaitAnalyzer.cpp
    src/Loader.cpp
    src/main.cpp
)

# Link OpenCV
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Define dataset path macro
add_definitions(-DDATASET_PATH=\"${CMAKE_SOURCE_DIR}/data/CASIA_B\")

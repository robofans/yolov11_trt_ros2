#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "yolov11.h"

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <image_path> <engine_path>" << std::endl;
    std::cout << "  image_path:  Path to input image" << std::endl;
    std::cout << "  engine_path: Path to TensorRT engine file" << std::endl;
}

int main(int argc, char* argv[]) {
    const std::string RED_COLOR = "\033[31m";
    const std::string GREEN_COLOR = "\033[32m";
    const std::string RESET_COLOR = "\033[0m";

    if (argc != 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string imagePath = argv[1];
    std::string enginePath = argv[2];

    try {
        YOLOv11 yolo(enginePath, logger);

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << RED_COLOR << "Failed to read image: " << imagePath << RESET_COLOR << std::endl;
            return 1;
        }

        yolo.preprocess(image);
        yolo.infer();

        std::vector<Detection> detections;
        yolo.postprocess(detections);
        yolo.draw(image, detections);

        std::string outputPath = "output_image.jpg";
        cv::imwrite(outputPath, image);
        std::cout << GREEN_COLOR << "Done. Output saved to " << outputPath << RESET_COLOR << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << RED_COLOR << "Error: " << e.what() << RESET_COLOR << std::endl;
        return 1;
    }

    return 0;
}
#pragma once

#include "NvInfer.h"
#include "macros.h"
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct Detection {
    float conf;
    int class_id;
    Rect bbox;
};

struct BuildConfig {
    bool fp16 = true;
    bool int8 = false;
    std::string calibDataPath = "";
    int32_t calibBatchSize = 32;
    bool useCalibCache = true;
    std::string calibCachePath = "";
    std::string layerPrecisionJson = "";
    int64_t workspaceSize = 1LL << 30;
};

class API YOLOv11 {
public:
    YOLOv11(string model_path, nvinfer1::ILogger& logger,
            const BuildConfig& buildConfig = BuildConfig());
    ~YOLOv11();
    void preprocess(Mat& image);
    void infer();
    void postprocess(vector<Detection>& output);
    void draw(Mat& image, const vector<Detection>& output);

    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;
    int input_w;
    int input_h;

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);
    float* gpu_buffers[2];
    float* cpu_output_buffer;
    cudaStream_t stream;
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    BuildConfig mBuildConfig;
    int num_detections;
    int detection_attribute_size;
    int num_classes = 80;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    vector<Scalar> colors;

    void build(std::string onnxPath, nvinfer1::ILogger& logger,
               const BuildConfig& config = BuildConfig());
    bool saveEngine(const std::string& filename);
};
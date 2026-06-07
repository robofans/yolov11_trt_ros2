#include "yolov11.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>
#include <numeric>

#define warmup true

YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger, const BuildConfig& buildConfig)
    : mBuildConfig(buildConfig)
{
    if (model_path.find(".onnx") == std::string::npos) {
        init(model_path, logger);
    } else {
        build(model_path, logger, buildConfig);
        saveEngine(model_path);
    }

#if NV_TENSORRT_MAJOR < 10
    auto input_dims = engine->getBindingDimensions(0);
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
#endif
}

void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger) {
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    context = engine->createExecutionContext();

#if NV_TENSORRT_MAJOR < 10
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
#else
    auto inputDims = engine->getTensorShape(engine->getIOTensorName(0));
    input_h = inputDims.d[2];
    input_w = inputDims.d[3];
    auto outputDims = engine->getTensorShape(engine->getIOTensorName(1));
    detection_attribute_size = outputDims.d[1];
    num_detections = outputDims.d[2];
    context->setTensorAddress(engine->getIOTensorName(0), gpu_buffers[0]);
    context->setTensorAddress(engine->getIOTensorName(1), gpu_buffers[1]);
#endif
    num_classes = detection_attribute_size - 4;

    cpu_output_buffer = new float[detection_attribute_size * num_detections];
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

    cuda_preprocess_init(MAX_IMAGE_SIZE);
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

YOLOv11::~YOLOv11() {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;
}

void YOLOv11::preprocess(Mat& image) {
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLOv11::infer() {
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    this->context->enqueueV3(this->stream);
#endif
}

void YOLOv11::postprocess(vector<Detection>& output) {
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1],
        num_detections * detection_attribute_size * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    for (int i = 0; i < det_output.cols; ++i) {
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        Point class_id_point;
        double score;
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold) {
            const float cx = det_output.at<float>(0, i);
            const float cy = det_output.at<float>(1, i);
            const float ow = det_output.at<float>(2, i);
            const float oh = det_output.at<float>(3, i);
            Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);
            confidences.push_back(score);
        }
    }

    vector<int> nms_result;
    {
        vector<float> areas(boxes.size());
        for (size_t i = 0; i < boxes.size(); i++)
            areas[i] = boxes[i].width * boxes[i].height;

        vector<int> sorted(boxes.size());
        std::iota(sorted.begin(), sorted.end(), 0);
        std::sort(sorted.begin(), sorted.end(),
            [&](int a, int b) { return confidences[a] > confidences[b]; });

        vector<bool> suppressed(boxes.size(), false);
        for (size_t i = 0; i < sorted.size(); i++) {
            if (suppressed[sorted[i]]) continue;
            nms_result.push_back(sorted[i]);
            for (size_t j = i + 1; j < sorted.size(); j++) {
                if (suppressed[sorted[j]]) continue;
                Rect inter = boxes[sorted[i]] & boxes[sorted[j]];
                float overlap = static_cast<float>(inter.area()) /
                    (areas[sorted[i]] + areas[sorted[j]] - inter.area());
                if (overlap > nms_threshold)
                    suppressed[sorted[j]] = true;
            }
        }
    }

    for (int i = 0; i < nms_result.size(); i++) {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = boxes[idx];
        output.push_back(result);
    }
}

void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger& logger, const BuildConfig& config) {
    auto builder = createInferBuilder(logger);
#if NV_TENSORRT_MAJOR < 10
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
#else
    INetworkDefinition* network = builder->createNetworkV2(0);
#endif
    IBuilderConfig* builderConfig = builder->createBuilderConfig();

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    if (!parsed) {
        std::cerr << "[YOLOv11] ERROR: Failed to parse ONNX model: " << onnxPath << std::endl;
        delete network;
        delete builderConfig;
        delete parser;
        return;
    }

    builderConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config.workspaceSize);

    if (config.fp16) {
        builderConfig->setFlag(BuilderFlag::kFP16);
        std::cout << "[YOLOv11] FP16 mode enabled" << std::endl;
    }

    if (config.int8) {
        builderConfig->setFlag(BuilderFlag::kINT8);
        std::cout << "[YOLOv11] INT8 mode enabled (requires Q/DQ nodes in ONNX)" << std::endl;
    }

    std::cout << "[YOLOv11] Building TensorRT engine..." << std::endl;
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *builderConfig) };
    if (!plan) {
        std::cerr << "[YOLOv11] ERROR: Failed to build serialized network" << std::endl;
        delete network;
        delete builderConfig;
        delete parser;
        return;
    }

    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    context = engine->createExecutionContext();

    delete network;
    delete builderConfig;
    delete parser;
    delete plan;

    std::cout << "[YOLOv11] Engine built successfully" << std::endl;
}

bool YOLOv11::saveEngine(const std::string& onnxpath) {
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos) {
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    } else {
        return false;
    }

    if (engine) {
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open()) {
            std::cout << "Create engine file " << engine_path << " failed" << std::endl;
            return false;
        }
        file.write((const char*)data->data(), data->size());
        file.close();
        delete data;
    }
    return true;
}

void YOLOv11::draw(Mat& image, const vector<Detection>& output) {
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    for (int i = 0; i < output.size(); i++) {
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        if (ratio_h > ratio_w) {
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        } else {
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }

        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, FILLED);
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}
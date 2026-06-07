#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <vision_msgs/msg/bounding_box2_d.hpp>
#include <cv_bridge/cv_bridge.h>

#include "yolov11.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

class YOLOv11PerceptionNode : public rclcpp::Node {
public:
    YOLOv11PerceptionNode() : Node("yolov11_perception_node") {
        this->declare_parameter<std::string>("engine_path", "");
        this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
        this->declare_parameter<double>("conf_threshold", 0.3);
        this->declare_parameter<double>("nms_threshold", 0.4);
        this->declare_parameter<bool>("enable_visualization", true);

        auto engine_path = this->get_parameter("engine_path").as_string();
        auto image_topic = this->get_parameter("image_topic").as_string();
        auto conf_thresh = this->get_parameter("conf_threshold").as_double();
        auto nms_thresh = this->get_parameter("nms_threshold").as_double();
        enable_viz_ = this->get_parameter("enable_visualization").as_bool();

        if (engine_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Required parameter 'engine_path' is empty.");
            throw std::runtime_error("engine_path is required");
        }

        RCLCPP_INFO(this->get_logger(), "Loading engine: %s", engine_path.c_str());
        yolo_ = std::make_unique<YOLOv11>(engine_path, trt_logger_);
        yolo_->conf_threshold = static_cast<float>(conf_thresh);
        yolo_->nms_threshold = static_cast<float>(nms_thresh);
        RCLCPP_INFO(this->get_logger(), "YOLOv11 loaded. Conf=%.2f NMS=%.2f",
                    yolo_->conf_threshold, yolo_->nms_threshold);

        det_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);
        det_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "detection_image", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());

        // keep_last(1) to only process latest frame, avoid queue buildup
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic, qos,
            std::bind(&YOLOv11PerceptionNode::imageCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Subscribed to '%s'", image_topic.c_str());
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat image;
        try {
            image = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        auto t0 = std::chrono::steady_clock::now();

        std::vector<Detection> detections;
        try {
            yolo_->preprocess(image);
            yolo_->infer();
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Preprocess/Infer error: %s", e.what());
            return;
        }

        yolo_->postprocess(detections);

        auto t1 = std::chrono::steady_clock::now();
        double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Rolling FPS
        frame_count_++;
        total_time_ms_ += dt_ms;
        if (frame_count_ >= fps_interval_) {
            fps_ = 1000.0 / (total_time_ms_ / frame_count_);
            frame_count_ = 0;
            total_time_ms_ = 0.0;
        }

        publishDetections(msg->header, detections);

        // Draw bboxes + FPS only when visualization is enabled
        if (enable_viz_) {
            cv::Mat display = image.clone();
            yolo_->draw(display, detections);

            std::string fps_text = cv::format("FPS: %.1f  |  %.1f ms", fps_, dt_ms);
            int baseline;
            cv::Size ts = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
            cv::rectangle(display, cv::Point(8, 8),
                          cv::Point(8 + ts.width + 8, 8 + ts.height + 8),
                          cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(display, fps_text, cv::Point(12, 12 + ts.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", display).toImageMsg();
            det_img_pub_->publish(*out_msg);
        }
    }

    void publishDetections(const std_msgs::msg::Header& header,
                           const std::vector<Detection>& detections) {
        vision_msgs::msg::Detection2DArray array_msg;
        array_msg.header = header;

        for (const auto& det : detections) {
            vision_msgs::msg::Detection2D d2d;
            d2d.header = header;

            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.class_id = std::to_string(det.class_id);
            hypothesis.hypothesis.score = det.conf;
            d2d.results.push_back(hypothesis);

            d2d.bbox.center.position.x = det.bbox.x + det.bbox.width / 2.0;
            d2d.bbox.center.position.y = det.bbox.y + det.bbox.height / 2.0;
            d2d.bbox.size_x = det.bbox.width;
            d2d.bbox.size_y = det.bbox.height;

            array_msg.detections.push_back(d2d);
        }

        det_pub_->publish(array_msg);
    }

    TrtLogger trt_logger_;
    std::unique_ptr<YOLOv11> yolo_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr det_img_pub_;

    bool enable_viz_ = true;
    int fps_interval_ = 10;
    int frame_count_ = 0;
    double total_time_ms_ = 0.0;
    double fps_ = 0.0;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<YOLOv11PerceptionNode>();
        rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 2);
        exec.add_node(node);
        exec.spin();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}

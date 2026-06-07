#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

static const char* COCO_CLASSES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

class YOLOv11VisualizationNode : public rclcpp::Node {
public:
    YOLOv11VisualizationNode() : Node("yolov11_visualization_node") {
        this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
        this->declare_parameter<std::string>("detections_topic", "/detections");
        this->declare_parameter<std::string>("output_topic", "/detection_image");
        this->declare_parameter<double>("text_scale", 0.5);
        this->declare_parameter<double>("text_thickness", 1.0);
        this->declare_parameter<int>("input_width", 640);
        this->declare_parameter<int>("input_height", 640);

        auto image_topic = this->get_parameter("image_topic").as_string();
        auto detections_topic = this->get_parameter("detections_topic").as_string();
        auto output_topic = this->get_parameter("output_topic").as_string();

        input_w_ = this->get_parameter("input_width").as_int();
        input_h_ = this->get_parameter("input_height").as_int();

        img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(output_topic, 10);

        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic, rclcpp::QoS(10).best_effort(),
            std::bind(&YOLOv11VisualizationNode::imageCallback, this, std::placeholders::_1));

        det_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
            detections_topic, 10,
            std::bind(&YOLOv11VisualizationNode::detectionsCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Visualization node started");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_image_ = msg;
    }

    void detectionsCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!latest_image_) return;

        cv::Mat image;
        try {
            image = cv_bridge::toCvShare(latest_image_, "bgr8")->image.clone();
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        auto text_scale = this->get_parameter("text_scale").as_double();
        auto text_thickness = this->get_parameter("text_thickness").as_double();

        for (const auto& d2d : msg->detections) {
            if (d2d.results.empty()) continue;

            int class_id = std::stoi(d2d.results[0].hypothesis.class_id);
            float score = d2d.results[0].hypothesis.score;

            float cx = d2d.bbox.center.position.x;
            float cy = d2d.bbox.center.position.y;
            float w = d2d.bbox.size_x;
            float h = d2d.bbox.size_y;
            int x1 = static_cast<int>(cx - w / 2.0f);
            int y1 = static_cast<int>(cy - h / 2.0f);
            int x2 = static_cast<int>(cx + w / 2.0f);
            int y2 = static_cast<int>(cy + h / 2.0f);

            mapBboxToImage(image.cols, image.rows, x1, y1, w, h);
            x2 = x1 + w;
            y2 = y1 + h;

            x1 = std::max(0, std::min(x1, image.cols - 1));
            y1 = std::max(0, std::min(y1, image.rows - 1));
            x2 = std::max(0, std::min(x2, image.cols - 1));
            y2 = std::max(0, std::min(y2, image.rows - 1));

            cv::Scalar color = getColor(class_id);

            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

            std::string label = (class_id >= 0 && class_id < 80)
                                      ? COCO_CLASSES[class_id]
                                      : std::to_string(class_id);
            label += " " + std::to_string(static_cast<int>(score * 100)) + "%";

            int baseline;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                  text_scale, static_cast<int>(text_thickness), &baseline);
            cv::rectangle(image, cv::Point(x1, y1 - text_size.height - 4),
                          cv::Point(x1 + text_size.width, y1), color, cv::FILLED);
            cv::putText(image, label, cv::Point(x1, y1 - 4),
                        cv::FONT_HERSHEY_SIMPLEX, text_scale, cv::Scalar(255, 255, 255),
                        static_cast<int>(text_thickness));
        }

        auto out_msg = cv_bridge::CvImage(latest_image_->header, "bgr8", image).toImageMsg();
        img_pub_->publish(*out_msg);
    }

    void mapBboxToImage(int img_w, int img_h, int& x, int& y, float& w, float& h) {
        float ratio_h = input_h_ / (float)img_h;
        float ratio_w = input_w_ / (float)img_w;

        if (ratio_h > ratio_w) {
            float pad = (input_h_ - ratio_w * img_h) / 2.0f;
            x = static_cast<int>(x / ratio_w);
            y = static_cast<int>((y - pad) / ratio_w);
            w = w / ratio_w;
            h = h / ratio_w;
        } else {
            float pad = (input_w_ - ratio_h * img_w) / 2.0f;
            x = static_cast<int>((x - pad) / ratio_h);
            y = static_cast<int>(y / ratio_h);
            w = w / ratio_h;
            h = h / ratio_h;
        }
    }

    static cv::Scalar getColor(int class_id) {
        static const cv::Scalar colors[] = {
            cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
            cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
            cv::Scalar(128, 0, 0), cv::Scalar(0, 128, 0), cv::Scalar(0, 0, 128),
            cv::Scalar(128, 128, 0), cv::Scalar(128, 0, 128), cv::Scalar(0, 128, 128),
        };
        return colors[class_id % 12];
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr det_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
    sensor_msgs::msg::Image::SharedPtr latest_image_;
    std::mutex mutex_;
    int input_w_;
    int input_h_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<YOLOv11VisualizationNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
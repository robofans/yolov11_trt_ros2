"""Launch YOLOv11 perception node + image_view.

Perception node draws bboxes + FPS directly on the image, published to /detection_image.
image_view shows the annotated image.

Usage:
  ENGINE_PATH=weights/yolo11n_int8.engine python3 perception_with_viz.launch.py
"""

import os
import sys

from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = os.environ.get(
        "YOLOV11TRT_DIR",
        os.path.join(os.path.dirname(__file__), "../../build"),
    )

    perception_exe = os.path.join(pkg_share, "yolov11_ros2_node")

    engine_path = os.environ.get(
        "ENGINE_PATH",
        os.path.join(os.getcwd(), "weights/yolo11n_head3_tail32_mixed.engine"),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "engine_path", default_value=engine_path,
            description="Path to TensorRT engine file"),
        DeclareLaunchArgument(
            "image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument(
            "conf_threshold", default_value="0.3"),
        DeclareLaunchArgument(
            "nms_threshold", default_value="0.4"),
        DeclareLaunchArgument(
            "enable_visualization", default_value="True",
            description="Draw bboxes+FPS on image and publish to /detection_image"),

        Node(
            executable=perception_exe,
            name="yolov11_perception",
            output="screen",
            parameters=[{
                "engine_path": LaunchConfiguration("engine_path"),
                "image_topic": LaunchConfiguration("image_topic"),
                "conf_threshold": LaunchConfiguration("conf_threshold"),
                "nms_threshold": LaunchConfiguration("nms_threshold"),
                "enable_visualization": LaunchConfiguration("enable_visualization"),
            }],
        ),

        Node(
            package="image_view",
            executable="image_view",
            name="detection_viewer",
            output="screen",
            remappings=[
                ("image", "/detection_image"),
            ],
        ),
    ])


def main(argv=sys.argv[1:]):
    ld = generate_launch_description()
    ls = LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == "__main__":
    main()

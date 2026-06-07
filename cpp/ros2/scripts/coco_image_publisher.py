#!/usr/bin/env python3
"""Publish COCO dataset images at a fixed rate for ROS2 perception testing.

Usage:
  python coco_image_publisher.py --image-dir datasets/coco/images/val2017
  python coco_image_publisher.py --image-dir datasets/coco/images/val2017 --rate 2 --topic /camera/image_raw
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CocoImagePublisher(Node):
    def __init__(self, image_dir, rate=2.0, topic="/camera/image_raw", loop=True,
                 shuffle=False, max_images=0):
        super().__init__("coco_image_publisher")

        self.pub = self.create_publisher(Image, topic, 10)
        self.bridge = CvBridge()
        self.loop = loop
        self.idx = 0

        extensions = (".jpg", ".jpeg", ".png", ".bmp")
        all_images = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ])

        if not all_images:
            self.get_logger().fatal(f"No images found in {image_dir}")
            raise FileNotFoundError(f"No images in {image_dir}")

        if max_images > 0 and max_images < len(all_images):
            all_images = all_images[:max_images]

        if shuffle:
            import random
            random.seed(42)
            random.shuffle(all_images)

        self.images = all_images
        interval = 1.0 / rate
        self.timer = self.create_timer(interval, self._publish)

        self.get_logger().info(
            f"Loaded {len(self.images)} images from {image_dir}, "
            f"publishing at {rate} Hz on '{topic}'"
        )

    def _publish(self):
        img_path = self.images[self.idx]
        img = cv2.imread(img_path)
        if img is None:
            self.get_logger().warn(f"Cannot read {img_path}, skipping")
            self._advance()
            return

        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.pub.publish(msg)
        self.get_logger().info(
            f"[{self.idx + 1}/{len(self.images)}] "
            f"{os.path.basename(img_path)}",
            throttle_duration_sec=5,
        )
        self._advance()

    def _advance(self):
        self.idx += 1
        if self.idx >= len(self.images):
            if self.loop:
                self.idx = 0
                self.get_logger().info("Looping back to start")
            else:
                self.get_logger().info("Finished, exiting")
                self.destroy_node()
                rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Publish COCO images to a ROS2 topic at a fixed rate"
    )
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing COCO images")
    parser.add_argument("--rate", type=float, default=2.0,
                        help="Publish rate in Hz (default: 2)")
    parser.add_argument("--topic", type=str, default="/camera/image_raw",
                        help="ROS2 topic to publish to")
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomize image order")
    parser.add_argument("--once", action="store_true",
                        help="Publish each image once then exit")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Limit to first N images")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Error: image directory not found: {args.image_dir}")
        sys.exit(1)

    rclpy.init()
    node = CocoImagePublisher(
        image_dir=args.image_dir,
        rate=args.rate,
        topic=args.topic,
        loop=not args.once,
        shuffle=args.shuffle,
        max_images=args.max_images,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

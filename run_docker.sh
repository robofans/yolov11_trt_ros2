#!/bin/bash

docker run --gpus all -it --rm --shm-size=16G \
  -v $(pwd):/workspace \
  -w /workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  nvcr.io/nvidia/pytorch/ros2:tensorrt8.6 \
  bash
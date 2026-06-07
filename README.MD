# ros_yolov11_trt

TensorRT-accelerated YOLOv11 object detection with Python tooling, C++ CUDA inference, and ROS2 Humble integration.

## Features

- PyTorch → ONNX model export
- TensorRT engine builder supporting FP32 / FP16 / INT8 / mixed precision (head + tail FP16)
- Speed vs. accuracy benchmarking
- Optional per-layer INT8 sensitivity analysis
- C++ CUDA-accelerated inference + ROS2 Humble perception node
- Tested on RTX 3050 laptop GPU

## Quick Start

### 1. Download Weights and Datasets

```bash
# Model weights
mkdir -p weights
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -O weights/yolo11n.pt
```

```bash
# COCO val2017 images (~6.2 GB)
mkdir -p datasets/coco/images
wget http://images.cocodataset.org/zips/val2017.zip -O datasets/val2017.zip
unzip -q datasets/val2017.zip -d datasets/coco/images && rm datasets/val2017.zip
```

```bash
# COCO val2017 annotations (YOLO-format txt, required for mAP evaluation)
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip -O datasets/coco2017labels.zip
unzip -qo datasets/coco2017labels.zip -d datasets && rm datasets/coco2017labels.zip
```

### 2. Prepare INT8 Calibration Data

```bash
python py/prepare_calib_data.py \
    --data datasets/coco_local.yaml \
    --num_samples 500 \
    --output_dir ./calib_data \
    --input_w 640 \
    --input_h 640
```

The dataset config `datasets/coco_local.yaml` is provided in the repo:

```yaml
path: datasets/coco
train: images/val2017
val: images/val2017
names:
  0: person
  1: bicycle
  ...
```

### 3. Export ONNX

```bash
python py/build_model_onnx.py --model weights/yolo11n.pt
```

### 4. Build TensorRT Engines

```bash
# FP32 engine
python py/build_trt_norm_engine.py weights/yolo11n.onnx --fp32

# FP16 engine
python py/build_trt_norm_engine.py weights/yolo11n.onnx --fp16

# INT8 engine (requires calibration data from step 2)
python py/build_trt_norm_engine.py weights/yolo11n.onnx --int8 --calib_dir calib_data

# Mixed precision: protect both head and tail layers
python py/build_trt_mixed_precision.py weights/yolo11n.onnx \
    --protect-head 3 --protect-tail 32 --calib-dir calib_data
```

### 5. Speed and Accuracy Benchmark

```bash
python py/compare_accuracy.py --data datasets/coco_local.yaml
```

Configure which engines to evaluate via the `MODELS` list in the script (defaults to FP32 / INT8 / Mixed).

Example output:

```
========================================================================
  Model                 FPS       ms  mAP@0.5      mAP   ΔmAP@0.5     ΔmAP
  ------------------------------------------------------------------------
  trt_fp32            257.9     3.88   0.5446   0.3879    +0.0000  +0.0000
  trt_int8            472.8     2.12   0.5348   0.3786    -0.0098  -0.0093
  trt_mixed           443.8     2.25   0.5393   0.3838    -0.0053  -0.0041
========================================================================
```

#### Sensitivity Analysis (Optional)

To generate per-layer precision configuration via PyTorch-level analysis:

```bash
python py/sensitivity_analysis.py \
    --model weights/yolo11n.pt \
    --data datasets/coco_local.yaml \
    --output configs/layer_precision.json
```

> Note: This simulates per-layer quantization in PyTorch, which may differ from actual TensorRT quantization. For production use, prefer `build_trt_mixed_precision.py --protect-tail N` for native TRT mixed precision.

## Precision Modes

| Mode  | Precision | Approx. FPS | Use Case |
|-------|-----------|-------------|----------|
| FP32  | Float32   | ~260        | Highest accuracy baseline |
| FP16  | Float16   | ~420        | Lossless acceleration |
| INT8  | Int8      | ~470        | Edge deployment |
| Mixed | INT8 + key layers FP16 | ~440 | <1% mAP drop, best speed-accuracy trade-off |

### INT8 Mixed Precision Strategy

The main source of INT8 accuracy degradation in detection tasks is the **detection head** (bbox regression + classification). Here's how to mitigate it:

- **Protect the last N layers**: `build_trt_mixed_precision.py --protect-tail N` — keep the last N Conv layers in FP16, quantize the rest to INT8
- **Protect both head and tail**: `--protect-head M --protect-tail N` — keep the first M Conv layers (shallow features) and last N Conv layers (detection head) in FP16
- Recommended range: for YOLO11n, N ≈ 32–64 Conv layers; tune based on measured accuracy

## Single-Image Inference (C++ CLI)

Quick smoke test to verify the engine works:

```bash
./cpp/build/YOLOv11TRT /path/to/image.jpg /path/to/engine.engine
```

## ROS2 Perception Node

### Build

```bash
source /opt/ros/humble/setup.bash
cd cpp && mkdir -p build && cd build
cmake .. -DBUILD_STANDALONE=OFF -DBUILD_ROS2=ON \
    -DTENSORRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
    -DTENSORRT_LIB_DIR=/usr/lib/x86_64-linux-gnu
make -j$(nproc)
```

Build artifacts:
- `yolov11_ros2_node` — Perception node (inference + bounding boxes + FPS overlay + `/detection_image` publisher)
- `yolov11_visualization_node` — Optional visualization node (subscribes to detections and draws boxes independently)

### Launch the Perception Pipeline

```bash
# Full visualization (default): boxes + FPS overlay + image_view window
python3 cpp/ros2/launch/perception_with_viz.launch.py

# Specify a custom engine
ENGINE_PATH=weights/yolo11n_mixed.engine python3 cpp/ros2/launch/perception_with_viz.launch.py

# Inference-only mode: disable visualization to skip clone + draw overhead
python3 cpp/ros2/launch/perception_with_viz.launch.py \
    enable_visualization:=False
```

Launch parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `engine_path` | `weights/yolo11n_int8.engine` | Engine path (can also be set via `ENGINE_PATH` env var) |
| `enable_visualization` | `True` | Enable boxes + FPS overlay + popup window; when disabled, only `/detections` is published |
| `image_topic` | `/camera/image_raw` | Subscribed image topic |
| `conf_threshold` | `0.3` | Confidence threshold |
| `nms_threshold` | `0.4` | NMS threshold |

### Publish COCO Test Images

```bash
python3 cpp/ros2/scripts/coco_image_publisher.py \
    --image-dir datasets/coco/images/val2017 \
    --rate 2
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image-dir` | (required) | COCO image directory |
| `--rate` | 2.0 | Publish rate in Hz |
| `--topic` | `/camera/image_raw` | ROS2 topic |
| `--shuffle` | false | Shuffle image order |
| `--once` | false | Exit after publishing all images once |
| `--max-images` | 0 (all) | Limit number of images |

### Topic Graph

```
/camera/image_raw ──→ perception_node ──→ /detection_image (bboxes + FPS, only when enable_visualization=True)
                           │                    ↓
                           │                image_view (GUI)
                           └──→ /detections (vision_msgs/Detection2DArray, always published)
```

Inference-only mode internal pipeline:

```
subscribe → cv_bridge → preprocess(GPU) → infer(TRT) → postprocess(NMS) → publish(/detections)
                          └────────── ~3ms for INT8 ──────────┘
```

## Dependencies

- Python 3.8+
- PyTorch (cu124)
- TensorRT 8.x / 10.x
- CUDA 12.4+
- OpenCV
- ROS2 Humble (optional)

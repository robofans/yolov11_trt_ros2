# ros_yolov11_trt

基于 TensorRT 的 YOLOv11 目标检测加速部署方案，支持 Python 导出与评测、C++ CUDA 推理以及 ROS2 Humble 感知节点。

## 项目特性

- PyTorch → ONNX 模型导出
- FP32 / FP16 / INT8 / 混合精度（头尾 FP16）TensorRT 引擎构建
- 速度与精度对比评测
- INT8 逐层敏感度分析（可选）
- C++ CUDA 加速推理 + ROS2 Humble 感知节点
- 测试环境：RTX 3050 笔记本 GPU

## 快速开始

### 1. 下载权重与数据集

```bash
# 模型权重
mkdir -p weights
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -O weights/yolo11n.pt
```

```bash
# COCO val2017 图片（约 6.2 GB）
mkdir -p datasets/coco/images
wget http://images.cocodataset.org/zips/val2017.zip -O datasets/val2017.zip
unzip -q datasets/val2017.zip -d datasets/coco/images && rm datasets/val2017.zip
```

```bash
# COCO val2017 标注（YOLO 格式 txt，mAP 评测必需）
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip -O datasets/coco2017labels.zip
unzip -qo datasets/coco2017labels.zip -d datasets && rm datasets/coco2017labels.zip
```

### 2. 准备 INT8 校准数据集

```bash
python py/prepare_calib_data.py \
    --data datasets/coco_local.yaml \
    --num_samples 500 \
    --output_dir ./calib_data \
    --input_w 640 \
    --input_h 640
```

数据集配置文件 `datasets/coco_local.yaml`（项目中已提供）：

```yaml
path: datasets/coco
train: images/val2017
val: images/val2017
names:
  0: person
  1: bicycle
  ...
```

### 3. 导出 ONNX

```bash
python py/build_model_onnx.py --model weights/yolo11n.pt
```

### 4. 构建 TensorRT 引擎

```bash
# FP32 引擎
python py/build_trt_norm_engine.py weights/yolo11n.onnx --fp32

# FP16 引擎
python py/build_trt_norm_engine.py weights/yolo11n.onnx --fp16

# INT8 引擎（需先准备校准数据）
python py/build_trt_norm_engine.py weights/yolo11n.onnx --int8 --calib_dir calib_data

# 混合精度（同时保护头尾）
python py/build_trt_mixed_precision.py weights/yolo11n.onnx \
    --protect-head 3 --protect-tail 32 --calib-dir calib_data
```

### 5. 速度与精度评测

```bash
python py/compare_accuracy.py --data datasets/coco_local.yaml
```

在脚本中通过 `MODELS` 列表配置待评测的引擎（默认对比 FP32 / INT8 / Mixed）。

输出示例：

```
========================================================================
  Model                 FPS       ms  mAP@0.5      mAP   ΔmAP@0.5     ΔmAP
  ------------------------------------------------------------------------
  trt_fp32            257.9     3.88   0.5446   0.3879    +0.0000  +0.0000
  trt_int8            472.8     2.12   0.5348   0.3786    -0.0098  -0.0093
  trt_mixed           443.8     2.25   0.5393   0.3838    -0.0053  -0.0041
========================================================================
```

#### 敏感度分析（可选）

如需基于 PyTorch 逐层分析生成层精度配置：

```bash
python py/sensitivity_analysis.py \
    --model weights/yolo11n.pt \
    --data datasets/coco_local.yaml \
    --output configs/layer_precision.json
```

> 注意：该分析在 PyTorch 层面模拟单层量化，与 TensorRT 实际量化存在偏差。推荐直接使用 `build_trt_mixed_precision.py --protect-tail N` 进行 TRT 原生的混合精度构建。

## 精度模式

| 模式 | 精度 | 参考 FPS | 适用场景 |
|------|------|----------|----------|
| FP32 | Float32 | ~260 | 最高精度基线 |
| FP16 | Float16 | ~420 | 精度无损加速 |
| INT8 | Int8 | ~470 | 边缘设备部署 |
| Mixed | INT8 + 关键层 FP16 | ~440 | INT8 掉点 <1%，兼顾精度与速度 |

### INT8 混合精度策略

检测任务的 INT8 量化精度损失主要来自**检测头**（bbox 回归 + 分类）。策略如下：

- **保护最后 N 层**：`build_trt_mixed_precision.py --protect-tail N`，将最后 N 个 Conv 层设为 FP16，其余 INT8
- **同时保护头尾**：`--protect-head M --protect-tail N`，前 M 个 Conv（浅层特征）和后 N 个 Conv（检测头）使用 FP16
- N 的推荐范围：YOLO11n 约 32~64 个 Conv 层，可根据实测精度调整

## 单图推理（C++ 命令行）

先验证引擎能否正常推理：

```bash
./cpp/build/YOLOv11TRT /path/to/image.jpg /path/to/engine.engine
```

## ROS2 感知节点

### 编译

```bash
source /opt/ros/humble/setup.bash
cd cpp && mkdir -p build && cd build
cmake .. -DBUILD_STANDALONE=OFF -DBUILD_ROS2=ON \
    -DTENSORRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
    -DTENSORRT_LIB_DIR=/usr/lib/x86_64-linux-gnu
make -j$(nproc)
```

编译产物：
- `yolov11_ros2_node` — 感知节点（推理 + 画框 + FPS + 发布 `/detection_image`）
- `yolov11_visualization_node` — 可视化节点（可选，独立订阅 detections 并画框）

### 启动感知管线

```bash
# 完整可视化（默认）：画框 + FPS + image_view 弹窗
python3 cpp/ros2/launch/perception_with_viz.launch.py

# 指定引擎
ENGINE_PATH=weights/yolo11n_mixed.engine python3 cpp/ros2/launch/perception_with_viz.launch.py

# 纯推理模式：关闭可视化，省去 clone + draw 开销
python3 cpp/ros2/launch/perception_with_viz.launch.py \
    enable_visualization:=False
```

启动参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `engine_path` | `weights/yolo11n_int8.engine` | 引擎路径（也可通过环境变量 `ENGINE_PATH` 设置） |
| `enable_visualization` | `True` | 开启画框 + FPS 叠加 + 弹窗；关闭后仅发布 `/detections` |
| `image_topic` | `/camera/image_raw` | 订阅的图像话题 |
| `conf_threshold` | `0.3` | 置信度阈值 |
| `nms_threshold` | `0.4` | NMS 阈值 |

### 发布 COCO 测试图像

```bash
python3 cpp/ros2/scripts/coco_image_publisher.py \
    --image-dir datasets/coco/images/val2017 \
    --rate 2
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image-dir` | (必需) | COCO 图片目录 |
| `--rate` | 2.0 | 发布频率 (Hz) |
| `--topic` | `/camera/image_raw` | ROS2 话题 |
| `--shuffle` | false | 随机打乱顺序 |
| `--once` | false | 发布一遍后退出 |
| `--max-images` | 0 (全部) | 限制图片数量 |

### 话题拓扑

```
/camera/image_raw ──→ perception_node ──→ /detection_image (bboxes + FPS, 仅 enable_visualization=True)
                           │                    ↓
                           │                image_view (GUI)
                           └──→ /detections (vision_msgs/Detection2DArray, 始终发布)
```

纯推理模式内部管线：

```
subscribe → cv_bridge → preprocess(GPU) → infer(TRT) → postprocess(NMS) → publish(/detections)
                          └────────── ~3ms for INT8 ──────────┘
```

## 依赖

- Python 3.8+
- PyTorch (cu124)
- TensorRT 8.x / 10.x
- CUDA 12.4+
- OpenCV
- ROS2 Humble（可选）

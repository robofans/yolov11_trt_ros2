# ros_yolov11_trt 工程完整复刻文档

## 1. 项目概述

本项目实现 YOLOv11 目标检测模型的 TensorRT 加速部署，包含：
- PyTorch 模型导出 ONNX
- 多种精度（FP32/FP16/INT8/混合精度）TensorRT 引擎构建
- 敏感性分析自动优化
- C++ CUDA 加速推理框架
- ROS2 Humble 感知节点

---

## 2. 目录结构

```
ros_yolov11_trt/
├── run_docker.sh                    # Docker 启动脚本
├── configs/
│   └── layer_precision.json         # 混合精度层精度配置
├── calib_data/                      # INT8 校准数据（需自行生成）
├── weights/                         # 模型权重（需自行下载）
├── python/
│   ├── trt/
│   │   ├── export_onnx.py
│   │   ├── build_trt_engine.py
│   │   ├── build_forward_mixed.py
│   │   ├── benchmark_trt.py
│   │   ├── check_precision_fast.py
│   │   └── check_precision_wrapper.sh
│   ├── analysis/
│   │   ├── sensitivity_analysis.py
│   │   ├── eval_mixed_precision.py
│   │   ├── benchmark_all.py
│   │   └── eval_int8_full.py
│   ├── data/
│   │   └── prepare_calib_data.py
│   └── models/
│       └── yolo11_model.py
├── cpp/
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── include/
│   │   ├── yolov11.h
│   │   ├── preprocess.h
│   │   ├── common.h
│   │   ├── cuda_utils.h
│   │   ├── logging.h
│   │   └── macros.h
│   ├── src/
│   │   ├── yolov11.cpp
│   │   └── preprocess.cu
│   └── ros2/
│       ├── perception_node.cpp
│       ├── visualization_node.cpp
│       ├── launch/
│       │   └── perception_with_viz.launch.py
│       └── scrip/
│           ├── test_image_pub.py
│           └── 1.png              # 测试图片（任意图片皆可）
└── datasets/
    └── coco_local.yaml
```

---

## 3. 数据集获取

### 3.1 COCO Val2017 数据集

COCO 2017 验证集用于精度评测和校准数据准备：

```bash
# 下载图片
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d datasets/

# 下载标注
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d datasets/

# 最终目录结构
datasets/
├── images/
│   └── val2017/          # 5000 张验证图片
├── annotations/
│   ├── instances_val2017.json
│   ├── captions_val2017.json
│   └── ...
```

### 3.2 COCO YAML 配置文件

创建 `datasets/coco_local.yaml`：

```yaml
# COCO 2017 dataset config for local path
path: /workspace/datasets
train: images/val2017
val: images/val2017

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```

---

## 4. 权重获取

### 4.1 YOLOv11n 预训练模型

```bash
mkdir -p weights
# 方法1: 从 Ultralytics 下载
python -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); print('Downloaded')"

# 方法2: 直接下载
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt -O weights/yolo11n.pt
```

---

## 5. 依赖安装

### 5.1 Python 环境

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics pycocotools tqdm polygraphy pyyaml opencv-python numpy
pip install pycuda
```

### 5.2 Docker 环境（TensorRT 构建用）

创建 `run_docker.sh`：

```bash
#!/bin/bash

docker run --gpus all -it --rm --shm-size=16G \
  -v $(pwd):/workspace \
  -w /workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  nvcr.io/nvidia/pytorch:23.10-py3-ros2 \
  bash
```

---

## 6. Python 完整源代码

### 6.1 python/trt/export_onnx.py

```python
#!/usr/bin/env python3
"""Export YOLO11 .pt → ONNX for TensorRT engine building."""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO11 to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Input .pt model path")
    parser.add_argument("--output", type=str, default="", help="Output .onnx path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    args = parser.parse_args()

    output = args.output
    if not output:
        stem = Path(args.model).stem
        output = str(Path(args.model).parent / f"{stem}.onnx")

    print(f"[Export] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[Export] Exporting to ONNX: {output}")
    print(f"         imgsz={args.imgsz}, batch={args.batch}, opset={args.opset}")

    success = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        half=False,
        simplify=True,
        dynamic=False,
    )

    if success:
        print(f"[Export] ONNX exported to: {output}")
        print(f"[Export] File size: {Path(output).stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"[Export] FAILED")


if __name__ == "__main__":
    main()
```

### 6.2 python/data/prepare_calib_data.py

```python
#!/usr/bin/env python3
"""
Calibration Data Preparation Tool
Prepare INT8 calibration data for TensorRT from COCO validation dataset.
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
              auto=False, scale_fill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def preprocess_image(img_path, input_w=640, input_h=640):
    """Preprocess single image: letterbox -> BGR2RGB -> normalize -> CHW."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img, ratio, pad = letterbox(img, (input_h, input_w), auto=False, scaleup=False)
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return img


def gather_coco_images(data_config, num_samples=500, split="val"):
    """Gather image paths from COCO dataset config."""
    import yaml
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Data config not found: {data_config}")
    with open(data_config, 'r') as f:
        cfg = yaml.safe_load(f)
    val_key = f"{split}" if split in cfg else "val"
    img_dir = cfg.get(val_key, "")
    if isinstance(img_dir, str):
        img_dir = img_dir
    elif isinstance(img_dir, list):
        img_dir = img_dir[0]
    if not os.path.isabs(img_dir):
        base_dir = os.path.dirname(os.path.abspath(data_config))
        img_dir = os.path.join(base_dir, img_dir)
    if not os.path.isdir(img_dir):
        alt_paths = [
            f"/home/lixiang/datasets/coco/{split}2017",
            f"/home/lixiang/datasets/coco/images/{split}2017",
            f"/datasets/coco/{split}2017",
            f"./coco/{split}2017",
            "/home/lixiang/work/test/yolo/datasets/images/val2017",
        ]
        found = False
        for p in alt_paths:
            if os.path.isdir(p):
                img_dir = p
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Cannot find COCO images directory. Tried: {img_dir}")
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    all_images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                         if f.lower().endswith(extensions)])
    if len(all_images) > num_samples:
        indices = np.linspace(0, len(all_images) - 1, num_samples, dtype=int)
        sampled = [all_images[i] for i in indices]
    else:
        sampled = all_images
        print(f"  Warning: only {len(sampled)} images available (requested {num_samples})")
    return sampled


def prepare_calibration_data(image_paths, output_dir, input_w=640, input_h=640, batch_size=32):
    """Output: calib_data.bin [N,3,H,W] float32, calib_data_shape.txt."""
    os.makedirs(output_dir, exist_ok=True)
    num_images = len(image_paths)
    output_shape = (num_images, 3, input_h, input_w)
    total_floats = num_images * 3 * input_h * input_w
    total_bytes = total_floats * 4
    bin_path = os.path.join(output_dir, "calib_data.bin")

    print(f"\n Preparing calibration data...")
    print(f"  Images: {num_images}")
    print(f"  Size:   {input_w}x{input_h}")
    print(f"  Output: {bin_path}")

    all_data = np.memmap(bin_path, dtype=np.float32, mode='w+', shape=output_shape)
    processed = 0
    for i in tqdm(range(0, num_images, batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        batch_size_actual = len(batch_paths)
        batch_data = np.zeros((batch_size_actual, 3, input_h, input_w), dtype=np.float32)
        for j, img_path in enumerate(batch_paths):
            try:
                batch_data[j] = preprocess_image(img_path, input_w, input_h)
            except Exception as e:
                print(f"  Warning: Failed to process {img_path}: {e}")
                batch_data[j] = np.full((3, input_h, input_w), 0.5, dtype=np.float32)
        all_data[i:i + batch_size_actual] = batch_data
        processed += batch_size_actual
    del all_data

    shape_path = os.path.join(output_dir, "calib_data_shape.txt")
    with open(shape_path, 'w') as f:
        f.write(f"{num_images} {3} {input_h} {input_w}\n")
        f.write(f"{batch_size}\n")

    list_path = os.path.join(output_dir, "image_list.txt")
    with open(list_path, 'w') as f:
        for p in image_paths:
            f.write(f"{p}\n")

    print(f"\n Done! Output at: {output_dir}")
    return bin_path


def main():
    parser = argparse.ArgumentParser(description="Prepare INT8 calibration data for TensorRT")
    parser.add_argument("--data", type=str, default="coco.yaml", help="Dataset config path")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of calibration images")
    parser.add_argument("--output_dir", type=str, default="./calib_data", help="Output directory")
    parser.add_argument("--input_w", type=int, default=640, help="Model input width")
    parser.add_argument("--input_h", type=int, default=640, help="Model input height")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    print("=" * 60)
    print(" TensorRT INT8 Calibration Data Preparation")
    print("=" * 60)

    print("[1/2] Gathering COCO validation images...")
    image_paths = gather_coco_images(args.data, args.num_samples)
    print(f"  Found {len(image_paths)} images")

    print("\n[2/2] Preprocessing and saving calibration data...")
    prepare_calibration_data(image_paths, args.output_dir, input_w=args.input_w,
                            input_h=args.input_h, batch_size=args.batch_size)

    print("\n Calibration data ready!")


if __name__ == "__main__":
    main()
```

### 6.3 python/analysis/sensitivity_analysis.py

```python
#!/usr/bin/env python3
"""
Sensitivity Analysis Tool for YOLOv11 PTQ
Analyze per-layer INT8 quantization sensitivity and generate layer precision config.
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

warnings.filterwarnings("ignore")


def quantize_tensor_per_tensor_absmax(x, num_bits=8):
    """Per-tensor symmetric INT8 quantization using absmax scaling."""
    scale = x.abs().max() + 1e-8
    qmax = 2 ** (num_bits - 1) - 1
    q_x = (x / scale * qmax).round().clamp(-qmax - 1, qmax)
    dq_x = q_x / qmax * scale
    return dq_x, scale


def quantize_tensor_per_channel_absmax(x, dim=0, num_bits=8):
    """Per-channel symmetric INT8 quantization for weights."""
    qmax = 2 ** (num_bits - 1) - 1
    shape = [1] * x.ndim
    shape[dim] = x.shape[dim]
    scales = x.abs().reshape(x.shape[0], -1).max(dim=1).values + 1e-8
    scales = scales.reshape(*shape)
    q_x = (x / scales * qmax).round().clamp(-qmax - 1, qmax)
    dq_x = q_x / qmax * scales
    return dq_x, scales


def simulate_conv_int8(weight, bias, input_fp32, conv_module):
    """Simulate INT8 quantization of a Conv2d layer."""
    w_int8, w_scale = quantize_tensor_per_channel_absmax(weight, dim=0)
    x_int8, x_scale = quantize_tensor_per_tensor_absmax(input_fp32)
    with torch.no_grad():
        output = nn.functional.conv2d(
            x_int8, w_int8, bias,
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            groups=conv_module.groups
        )
    return output


class LayerOutputHook:
    """Register forward hooks to capture layer inputs and outputs."""

    def __init__(self):
        self.fp32_outputs = {}
        self.quant_outputs = {}
        self.inputs = {}
        self.handles = []

    def _fp32_hook_fn(self, name):
        def hook(module, input, output):
            self.fp32_outputs[name] = output.detach()
            self.inputs[name] = input[0].detach()
        return hook

    def register_fp32_hooks(self, model, conv_layers_dict):
        for name in conv_layers_dict:
            mod = conv_layers_dict[name]
            handle = mod.register_forward_hook(self._fp32_hook_fn(name))
            self.handles.append(handle)

    def register_quant_hooks(self, model, conv_layers_dict, target_layer_name):
        self.quant_outputs.clear()
        def _quant_hook_fn(name):
            def hook(module, input, output):
                if name == target_layer_name:
                    w = module.weight
                    b = module.bias
                    if b is None:
                        b = torch.zeros(w.shape[0], device=w.device)
                    q_out = simulate_conv_int8(w, b, input[0].detach(), module)
                    self.quant_outputs[name] = q_out.detach()
                    self.fp32_outputs[name] = output.detach()
            return hook
        handles = []
        for name in conv_layers_dict:
            mod = conv_layers_dict[name]
            h = mod.register_forward_hook(_quant_hook_fn(name))
            handles.append(h)
        return handles

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def cosine_similarity(tensor_a, tensor_b):
    """Compute cosine similarity between two flattened tensors."""
    a = tensor_a.flatten()
    b = tensor_b.flatten()
    if a.norm() < 1e-8 or b.norm() < 1e-8:
        return 1.0 if (a - b).norm() < 1e-8 else 0.0
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))


def sqnr(tensor_a, tensor_b):
    """Signal-to-Quantization-Noise Ratio in dB."""
    signal_power = (tensor_a ** 2).mean() + 1e-8
    noise_power = ((tensor_a - tensor_b) ** 2).mean() + 1e-8
    return float(10 * torch.log10(signal_power / noise_power))


def relative_mse(tensor_a, tensor_b):
    """Relative MSE between two tensors."""
    mse = ((tensor_a - tensor_b) ** 2).mean()
    var = (tensor_a ** 2).mean() + 1e-8
    return float((mse / var).sqrt())


def compute_layer_metric(fp32_out, quant_out, metric="cosine"):
    """Compute sensitivity metric between FP32 and quantized outputs."""
    if metric == "cosine":
        return cosine_similarity(fp32_out, quant_out)
    elif metric == "sqnr":
        return sqnr(fp32_out, quant_out)
    elif metric == "mse":
        return relative_mse(fp32_out, quant_out)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def collect_conv_layers(model):
    """Collect all Conv2d layers with their names from a YOLO model."""
    conv_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers[name] = module
    return conv_layers


def analyze_sensitivity_per_layer(model, dataloader, conv_layers,
                                   num_samples=500, metric="cosine", device="cuda"):
    """Analyze per-layer INT8 quantization sensitivity."""
    model.to(device)
    model.eval()
    layer_scores = {name: [] for name in conv_layers}
    layer_names = list(conv_layers.keys())
    n_layers = len(layer_names)
    hook_system = LayerOutputHook()

    print(f"\n Analyzing {n_layers} Conv2d layers across {num_samples} samples...")
    samples_processed = 0

    for batch_idx, batch in enumerate(dataloader):
        if samples_processed >= num_samples:
            break
        if isinstance(batch, dict):
            images = batch['img']
        else:
            images = batch[0]
        images = images.to(device)
        batch_size = images.shape[0]
        remaining = min(batch_size, num_samples - samples_processed)
        samples_processed += remaining

        if batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx}... ({samples_processed}/{num_samples})")

        fp32_hooks = []
        for name in layer_names:
            mod = conv_layers[name]
            h = mod.register_forward_hook(hook_system._fp32_hook_fn(name))
            fp32_hooks.append(h)

        with torch.no_grad():
            hook_system.fp32_outputs.clear()
            hook_system.inputs.clear()
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            _ = model(images)

        for name in layer_names:
            if name not in hook_system.fp32_outputs:
                continue
            fp32_out = hook_system.fp32_outputs[name]
            mod = conv_layers[name]
            inp = hook_system.inputs.get(name)
            if inp is None:
                continue
            with torch.no_grad():
                quant_out = simulate_conv_int8(mod.weight, mod.bias, inp, mod)
            if quant_out.shape != fp32_out.shape:
                quant_out = nn.functional.interpolate(
                    quant_out, size=fp32_out.shape[2:], mode='bilinear', align_corners=False
                )
            score = compute_layer_metric(
                fp32_out[:remaining].cpu(),
                quant_out[:remaining].cpu(),
                metric
            )
            layer_scores[name].append(score)

        for h in fp32_hooks:
            h.remove()

    aggregated = {}
    for name, scores in layer_scores.items():
        if scores:
            aggregated[name] = float(np.mean(scores))
        else:
            aggregated[name] = 1.0

    if metric in ("cosine", "sqnr"):
        sorted_layers = sorted(aggregated.items(), key=lambda x: x[1])
    else:
        sorted_layers = sorted(aggregated.items(), key=lambda x: -x[1])

    return sorted_layers


def generate_layer_precision_config(sensitivity_results, baseline_score=1.0,
                                     sensitivity_threshold=0.05, metric="cosine",
                                     model_name="yolo11n", baseline_mAP=0.0):
    """Generate layer precision config JSON from sensitivity analysis results."""
    if metric == "cosine":
        sensitive_layers = [
            name for name, score in sensitivity_results
            if score < (baseline_score - sensitivity_threshold)
        ]
    elif metric == "sqnr":
        sensitive_layers = [
            name for name, score in sensitivity_results
            if score < sensitivity_threshold
        ]
    elif metric == "mse":
        sensitive_layers = [
            name for name, score in sensitivity_results
            if score > sensitivity_threshold
        ]
    else:
        sensitive_layers = []

    config = {
        "model": model_name,
        "sensitivity_analysis_method": f"per_layer_{metric}",
        "baseline_score": baseline_score,
        "sensitivity_threshold": sensitivity_threshold,
        "baseline_mAP": baseline_mAP,
        "total_layers": len(sensitivity_results),
        "sensitive_layers_count": len(sensitive_layers),
        "default_precision": "INT8",
        "sensitive_layers": sensitive_layers,
        "sensitive_precision": "FP16",
        "layer_details": [
            {
                "name": name,
                "score": round(score, 6),
                "precision": "FP16" if name in sensitive_layers else "INT8"
            }
            for name, score in sensitivity_results
        ]
    }
    return config


def detect_head_layer_names(conv_layers):
    """Categorize Conv2d layers into Backbone / Neck / Head."""
    categories = {"backbone": [], "neck": [], "head": [], "other": []}
    for name in conv_layers:
        if "model.23" in name or "dfl" in name:
            categories["head"].append(name)
        elif any(f"model.{i}" in name for i in range(0, 11)):
            categories["backbone"].append(name)
        elif any(f"model.{i}" in name for i in range(11, 20)):
            categories["neck"].append(name)
        else:
            categories["other"].append(name)
    return categories


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Per-Layer INT8 Sensitivity Analysis")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--data", type=str, default="coco8.yaml")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "sqnr", "mse"])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="configs/layer_precision.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--head_fp16", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 70)
    print(" YOLOv11 INT8 Sensitivity Analysis")
    print("=" * 70)

    print("\n[1/4] Loading model...")
    model = YOLO(args.model)
    pt_model = model.model
    print(f"  Found {len(pt_model.model)} sequential modules")

    print("\n[2/4] Collecting Conv2d layers...")
    conv_layers = collect_conv_layers(pt_model)
    print(f"  Found {len(conv_layers)} Conv2d layers")

    categories = detect_head_layer_names(conv_layers)
    for cat, layers in categories.items():
        print(f"    {cat}: {len(layers)} layers")

    print("\n[3/4] Running sensitivity analysis...")

    from ultralytics.data import build_yolo_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    args_cfg = get_cfg(DEFAULT_CFG)
    args_cfg.model = args.model
    args_cfg.data = args.data
    args_cfg.imgsz = args.imgsz
    args_cfg.batch = args.batch_size
    args_cfg.mode = "val"

    data_dict = check_det_dataset(args.data)
    img_path = data_dict['val']
    dataset = build_yolo_dataset(args_cfg, img_path, args.batch_size, data_dict,
                                  mode="val", rect=False, stride=32)
    dataloader = build_dataloader(dataset, batch=args.batch_size, workers=4,
                                   shuffle=False, rank=-1)

    sensitivity_results = analyze_sensitivity_per_layer(
        pt_model, dataloader, conv_layers,
        num_samples=args.num_samples, metric=args.metric, device=args.device
    )

    print(f"\n  Sensitivity analysis complete!")
    print(f"  Top-10 most sensitive layers:")
    for name, score in sensitivity_results[:10]:
        print(f"    {name:50s} score={score:.6f}")

    print(f"\n[4/4] Generating layer precision config...")

    all_scores = [s for _, s in sensitivity_results]
    if args.metric == "cosine":
        baseline_score = float(np.median(all_scores))
    elif args.metric == "sqnr":
        baseline_score = float(np.median(all_scores))

    config = generate_layer_precision_config(
        sensitivity_results,
        baseline_score=baseline_score,
        sensitivity_threshold=args.threshold,
        metric=args.metric,
        model_name=Path(args.model).stem,
        baseline_mAP=0.0,
    )

    if args.head_fp16:
        for detail in config["layer_details"]:
            if any(x in detail["name"] for x in ["model.23", "dfl"]):
                if detail["precision"] == "INT8":
                    detail["precision"] = "FP16"
                    config["sensitive_layers"].append(detail["name"])
                    config["sensitive_layers_count"] += 1

    config["sensitive_layers"] = list(set(config["sensitive_layers"]))
    config["sensitive_layers_count"] = len(config["sensitive_layers"])

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n Config saved to: {args.output}")
    print(f"  INT8 layers: {config['total_layers'] - config['sensitive_layers_count']}")
    print(f"  FP16 layers: {config['sensitive_layers_count']}")


if __name__ == "__main__":
    main()
```

### 6.4 python/trt/build_trt_engine.py

```python
#!/usr/bin/env python3
"""Build TensorRT engines (FP32/FP16/INT8/Mixed) from ONNX for YOLOv11."""

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np

os.environ["TRT_CASK_DISABLE"] = "1"

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_calib_data(calib_dir, batch_size=32):
    """Load preprocessed calibration data from prepare_calib_data.py output."""
    shape_path = os.path.join(calib_dir, "calib_data_shape.txt")
    bin_path = os.path.join(calib_dir, "calib_data.bin")
    if not os.path.exists(shape_path) or not os.path.exists(bin_path):
        raise FileNotFoundError(f"Calibration data not found in {calib_dir}.")
    with open(shape_path) as f:
        n, c, h, w = [int(x) for x in f.readline().strip().split()]
        batch_line = f.readline().strip()
        calib_batch = int(batch_line) if batch_line else batch_size
    data = np.fromfile(bin_path, dtype=np.float32).reshape(n, c, h, w)
    return data, calib_batch


class Calibrator(trt.IInt8MinMaxCalibrator):
    """INT8 calibrator loading preprocessed data from .bin file."""

    def __init__(self, calib_dir, batch_size=32, cache_file="calib.cache"):
        super().__init__()
        data, calib_batch = load_calib_data(calib_dir, batch_size)
        self._data = data
        self._batch_size = calib_batch
        self._n = data.shape[0]
        self._current_idx = 0
        self._cache_file = os.path.join(calib_dir, cache_file)
        self._device_input = cuda.mem_alloc(
            self._batch_size * int(np.prod(data.shape[1:])) * 4
        )

    def get_batch_size(self):
        return self._batch_size

    def get_batch(self, names):
        if self._current_idx >= self._n:
            return None
        batch_size = min(self._batch_size, self._n - self._current_idx)
        batch = np.ascontiguousarray(
            self._data[self._current_idx:self._current_idx + batch_size]
        )
        cuda.memcpy_htod(self._device_input, batch)
        self._current_idx += batch_size
        return [int(self._device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self._cache_file):
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self._cache_file, "wb") as f:
            f.write(cache)


def load_layer_precision(json_path):
    """Load layer precision config and return {layer_name: trt.DataType}."""
    with open(json_path) as f:
        config = json.load(f)
    precision_map = {"INT8": trt.DataType.INT8, "FP16": trt.DataType.HALF,
                     "FP32": trt.DataType.FLOAT}
    layer_map = {}
    for detail in config.get("layer_details", []):
        name = detail["name"]
        prec_str = detail["precision"]
        if prec_str in precision_map:
            layer_map[name] = precision_map[prec_str]
    default_prec = precision_map.get(config.get("default_precision", "INT8"), trt.DataType.INT8)
    return layer_map, default_prec, config.get("sensitive_layers", [])


def _get_name_segments(name):
    """Split a layer name into significant segments for matching."""
    return set(name.replace(".", "/").strip("/").split("/"))


def apply_layer_precision(network, layer_map, default_prec):
    """Apply per-layer precision constraints to the network."""
    int8_count = 0
    fp16_count = 0
    fp32_count = 0
    matched_names = set()
    map_segments = {name: _get_name_segments(name) for name in layer_map}

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type != trt.LayerType.CONVOLUTION:
            continue
        name = layer.name
        trt_segments = _get_name_segments(name)

        matched_key = None
        for key, segs in map_segments.items():
            if segs.issubset(trt_segments):
                matched_key = key
                matched_names.add(key)
                break

        if matched_key is None:
            continue

        prec = layer_map[matched_key]
        if prec == default_prec:
            continue

        layer.precision = prec
        for j in range(layer.num_outputs):
            layer.set_output_type(j, prec)

        if prec == trt.DataType.INT8:
            int8_count += 1
        elif prec == trt.DataType.HALF:
            fp16_count += 1
        elif prec == trt.DataType.FLOAT:
            fp32_count += 1

    print(f"  Layer precision applied: INT8={int8_count}, FP16={fp16_count}, FP32={fp32_count}")
    unmatched = set(layer_map.keys()) - matched_names
    if unmatched:
        print(f"  Unmatched layer_map entries: {len(unmatched)}")
    return int8_count, fp16_count, fp32_count


def build_engine(onnx_path, output_path, fp16=False, int8=False,
                 calib_dir=None, layer_precision=None,
                 workspace_size=1 << 30, batch_size=1):
    """Build TensorRT engine from ONNX with specified precision settings."""
    onnx_path = os.path.abspath(onnx_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[Build] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"  Parse error {err}: {parser.get_error(err)}")
            raise RuntimeError("ONNX parse failed")

    print(f"  Network: {network.num_layers} layers")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    if fp16:
        print("[Build] FP16 mode enabled")
        config.set_flag(trt.BuilderFlag.FP16)

    calibrator = None
    if int8:
        print("[Build] INT8 mode enabled")
        config.set_flag(trt.BuilderFlag.INT8)
        if calib_dir and os.path.isdir(calib_dir):
            calibrator = Calibrator(calib_dir, batch_size=32)
            config.int8_calibrator = calibrator
            print(f"  Calibrator initialized: {calib_dir}")
        if layer_precision:
            print(f"[Build] Loading layer precision: {layer_precision}")
            layer_map, default_prec, sensitive = load_layer_precision(layer_precision)
            print(f"  Default precision: {'INT8' if default_prec == trt.DataType.INT8 else 'FP16'}")
            print(f"  Sensitive layers: {len(sensitive)} (FP16)")
            apply_layer_precision(network, layer_map, default_prec)

    print("[Build] Building engine (may take a while)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(output_path, "wb") as f:
        f.write(serialized)

    dt = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[Build] Done! ({dt:.1f}s, {size_mb:.1f} MB)")
    print(f"  Output: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("onnx", type=str, help="Input ONNX model path")
    parser.add_argument("--output", "-o", type=str, default="", help="Output .engine path")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--calib_dir", type=str, default="", help="Calibration data directory")
    parser.add_argument("--layer_precision", type=str, default="", help="Layer precision JSON")
    parser.add_argument("--workspace", type=int, default=1 << 30, help="Workspace size (bytes)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    if not args.output:
        stem = Path(args.onnx).stem
        prec = "fp32"
        if args.int8 and args.layer_precision:
            prec = "mixed"
        elif args.int8:
            prec = "int8"
        elif args.fp16:
            prec = "fp16"
        args.output = str(Path(args.onnx).parent / f"{stem}_{prec}.engine")

    build_engine(
        onnx_path=args.onnx, output_path=args.output,
        fp16=args.fp16, int8=args.int8,
        calib_dir=args.calib_dir, layer_precision=args.layer_precision,
        workspace_size=args.workspace, batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
```

### 6.5 python/trt/build_forward_mixed.py

```python
#!/usr/bin/env python3
"""Build a mixed-precision TRT engine with FP16 and INT8 layers, or evaluate existing engine."""

import argparse, os, time, sys
import numpy as np
os.environ["TRT_CASK_DISABLE"] = "1"
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_PATH = "weights/yolo11n.onnx"
CALIB_CACHE = "calib_data/calib_minmax.cache"


class Calibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, cache_file):
        super().__init__()
        self._cache_file = cache_file
        self._dummy = cuda.mem_alloc(1 * 3 * 640 * 640 * 4)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        return None

    def read_calibration_cache(self):
        if os.path.exists(self._cache_file):
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        pass


def build_engine(fp16_layers, output_path, reverse=False):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[Build] Parsing ONNX: {ONNX_PATH}")
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"  Parse error {err}: {parser.get_error(err)}")
            raise RuntimeError("ONNX parse failed")

    n_layers = network.num_layers
    print(f"  Network: {n_layers} layers total")
    print(f"  Direction: {'reverse (last N layers)' if reverse else 'forward (first N layers)'}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 31)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    constrained = 0
    if reverse:
        start = max(0, n_layers - fp16_layers)
        boundary_start = max(0, start - 1)

        for i in range(boundary_start, n_layers):
            layer = network.get_layer(i)
            prev_layer = network.get_layer(i - 1) if i > 0 else None

            try:
                layer.precision = trt.DataType.HALF
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.HALF)
                if prev_layer is not None:
                    try:
                        prev_layer.precision = trt.DataType.HALF
                        for j in range(prev_layer.num_outputs):
                            prev_layer.set_output_type(j, trt.DataType.HALF)
                    except Exception:
                        pass
                constrained += 1
            except Exception:
                pass
    else:
        for i in range(min(fp16_layers, n_layers)):
            layer = network.get_layer(i)
            try:
                layer.precision = trt.DataType.HALF
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.HALF)
                constrained += 1
            except Exception:
                pass

    print(f"  FP16 constrained: {constrained}/{fp16_layers} layers (remaining use INT8)")

    calibrator = Calibrator(CALIB_CACHE)
    config.int8_calibrator = calibrator

    print(f"[Build] Building engine...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    with open(output_path, "wb") as f:
        f.write(serialized)

    dt = time.time() - t0
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[Build] Done! ({dt:.1f}s, {size_mb:.1f} MB) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build mixed-precision engine: first/last N layers FP16, rest INT8")
    parser.add_argument("--fp16-layers", type=int, help="Number of layers to run in FP16")
    parser.add_argument("--reverse", action="store_true", help="Protect last N layers instead of first N")
    parser.add_argument("--output", "-o", type=str, default="", help="Output engine path")
    parser.add_argument("--check", action="store_true", help="Run accuracy check after building")
    parser.add_argument("--engine", "-e", type=str, default="", help="Existing engine path to evaluate")
    args = parser.parse_args()

    if args.engine:
        if not os.path.exists(args.engine):
            print(f"Error: Engine file not found: {args.engine}")
            sys.exit(1)
        args.output = args.engine
        print(f"[Eval] Using existing engine: {args.engine}")
    elif args.fp16_layers is None:
        parser.error("Either --fp16-layers or --engine is required")
    else:
        if not args.output:
            suffix = "reverse" if args.reverse else "forward"
            args.output = f"weights/yolo11n_{suffix}_mixed.engine"
        print(f"[Build] fp16_layers={args.fp16_layers}, reverse={args.reverse}")
        build_engine(args.fp16_layers, args.output, args.reverse)

    if args.check:
        import subprocess
        import shutil
        shutil.copy(args.output, "polygraphy_debug.engine")

        result = subprocess.run(["python3", "python/trt/check_precision_fast.py"],
                                capture_output=True, text=True, errors="replace")
        print(result.stdout)

        print("\n[Benchmark] Running speed test...")
        engine_path = args.output
        benchmark_script = (
            "import time, os, sys\n"
            "os.environ['CUDA_MODULE_LOADING'] = 'LAZY'\n"
            "import numpy as np\n"
            "import pycuda.driver as cuda\n"
            "import pycuda.autoinit\n"
            "import tensorrt as trt\n"
            "logger = trt.Logger(trt.Logger.WARNING)\n"
            "with open('{}', 'rb') as f:\n"
            "    runtime = trt.Runtime(logger)\n"
            "    engine = runtime.deserialize_cuda_engine(f.read())\n"
            "context = engine.create_execution_context()\n"
            "input_name = engine.get_tensor_name(0)\n"
            "output_name = engine.get_tensor_name(1)\n"
            "context.set_input_shape(input_name, (1, 3, 640, 640))\n"
            "input_size = 1 * 3 * 640 * 640 * 4\n"
            "output_size = 1 * 84 * 8400 * 4\n"
            "d_input = cuda.mem_alloc(input_size)\n"
            "d_output = cuda.mem_alloc(output_size)\n"
            "stream = cuda.Stream()\n"
            "context.set_tensor_address(input_name, int(d_input))\n"
            "context.set_tensor_address(output_name, int(d_output))\n"
            "dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)\n"
            "for _ in range(10):\n"
            "    cuda.memcpy_htod_async(d_input, dummy, stream)\n"
            "    context.execute_async_v3(stream.handle)\n"
            "stream.synchronize()\n"
            "N = 100\n"
            "t0 = time.perf_counter()\n"
            "for _ in range(N):\n"
            "    cuda.memcpy_htod_async(d_input, dummy, stream)\n"
            "    context.execute_async_v3(stream.handle)\n"
            "    stream.synchronize()\n"
            "elapsed = time.perf_counter() - t0\n"
            "fps = N / elapsed\n"
            "print('  FPS: %.1f (%d runs in %.3fs)' % (fps, N, elapsed))\n"
        ).format(engine_path)

        speed_result = subprocess.run(["python3", "-c", benchmark_script],
                                capture_output=True, text=True)
        if speed_result.stdout:
            print(speed_result.stdout)
        else:
            print("  Speed test error:", speed_result.stderr[:200] if speed_result.stderr else "unknown")

        if result.returncode == 0:
            print("[Check] PASS")
        else:
            print("[Check] FAIL (mAP below threshold)")
        os.remove("polygraphy_debug.engine")


if __name__ == "__main__":
    main()
```

### 6.6 python/trt/benchmark_trt.py

```python
#!/usr/bin/env python3
"""TensorRT YOLOv11 Speed + Accuracy Benchmark."""

import os
os.environ["TRT_CASK_DISABLE"] = "1"

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):
    """Deserialize TensorRT engine from file."""
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def benchmark_speed_trt(engine, n_warmup=50, n_measured=200, batch=1):
    """Measure inference speed of a TensorRT engine."""
    import pycuda.driver as cuda
    import pycuda.autoinit

    ctx = engine.create_execution_context()
    input_idx = 0
    output_idx = 1
    input_name = engine.get_tensor_name(input_idx)
    output_name = engine.get_tensor_name(output_idx)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    if -1 in input_shape:
        input_shape = (batch, 3, 640, 640)
        ctx.set_input_shape(input_name, input_shape)
    if -1 in output_shape:
        output_shape = (1, 84, 8400)

    input_size = int(np.prod(input_shape)) * 4
    output_size = int(np.prod(output_shape)) * 4
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    dummy = np.random.randn(*input_shape).astype(np.float32)
    stream = cuda.Stream()
    ctx.set_tensor_address(input_name, int(d_input))
    ctx.set_tensor_address(output_name, int(d_output))

    cuda.memcpy_htod_async(d_input, dummy, stream)
    for _ in range(n_warmup):
        ctx.execute_async_v3(stream.handle)
    stream.synchronize()

    times = []
    for _ in range(n_measured):
        cuda.memcpy_htod_async(d_input, dummy, stream)
        start = time.perf_counter()
        ctx.execute_async_v3(stream.handle)
        stream.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)
    mean_ms = float(times.mean())
    std_ms = float(times.std())
    fps = 1000.0 / mean_ms * batch
    return mean_ms, std_ms, fps


def benchmark_accuracy_trt(engine_path, data_path, imgsz=640, batch=32, device="0"):
    """Evaluate mAP of a TensorRT engine using Ultralytics YOLO."""
    from ultralytics import YOLO
    model = YOLO(engine_path)
    results = model.val(data=data_path, imgsz=imgsz, batch=batch,
                        device=device, verbose=False, max_det=300)
    return {
        "map50": float(results.box.map50),
        "map": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }


def get_engine_path(onnx_path, suffix):
    stem = Path(onnx_path).stem
    return str(Path(onnx_path).parent / f"{stem}_{suffix}.engine")


def main():
    parser = argparse.ArgumentParser(description="TRT YOLOv11 Speed + Accuracy Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model path (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Dataset config (YAML)")
    parser.add_argument("--calib_dir", type=str, required=True, help="Calibration data dir")
    parser.add_argument("--layer_precision", type=str, required=True, help="Layer precision JSON")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--eval_batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output", type=str, default="results/trt_benchmark_results.json")
    parser.add_argument("--n_warmup", type=int, default=50)
    parser.add_argument("--n_measured", type=int, default=200)
    parser.add_argument("--speed_only", action="store_true")
    parser.add_argument("--accuracy_only", action="store_true")
    parser.add_argument("--keep_engines", action="store_true")
    args = parser.parse_args()

    do_accuracy = not args.speed_only
    do_speed = not args.accuracy_only

    onnx_path = str(Path(args.model).parent / f"{Path(args.model).stem}.onnx")
    if not os.path.exists(onnx_path):
        print("=" * 60)
        print("[0/4] Exporting ONNX...")
        print("=" * 60)
        subprocess.check_call([
            sys.executable, "python/trt/export_onnx.py",
            "--model", args.model, "--output", onnx_path,
        ])

    configs = {
        "fp32": {"output": get_engine_path(onnx_path, "fp32"), "fp16": False, "int8": False},
        "fp16": {"output": get_engine_path(onnx_path, "fp16"), "fp16": True, "int8": False},
        "int8_full": {"output": get_engine_path(onnx_path, "int8"), "fp16": True, "int8": True, "layers": None},
        "mixed": {"output": get_engine_path(onnx_path, "mixed"), "fp16": True, "int8": True, "layers": args.layer_precision},
    }

    results = {
        "model": str(Path(args.model).stem),
        "config": {"imgsz": args.imgsz, "speed_batch": args.batch, "eval_batch": args.eval_batch},
        "engines": {}, "speed": {}, "accuracy": {},
    }

    for cfg_name, cfg in configs.items():
        print(f"\n{'=' * 60}\n[Benchmark] {cfg_name.upper()}\n{'=' * 60}")

        if not os.path.exists(cfg["output"]):
            print(f"  Building engine: {cfg['output']}")
            cmd = [sys.executable, "python/trt/build_trt_engine.py", onnx_path,
                   "--output", cfg["output"]]
            if cfg["fp16"]:
                cmd.append("--fp16")
            if cfg["int8"]:
                cmd.append("--int8")
                cmd.extend(["--calib_dir", args.calib_dir])
                if cfg.get("layers"):
                    cmd.extend(["--layer_precision", cfg["layers"]])
            subprocess.check_call(cmd)

        engine_size_mb = os.path.getsize(cfg["output"]) / 1024 / 1024
        results["engines"][cfg_name] = {"path": cfg["output"], "size_mb": round(engine_size_mb, 1)}

        if do_speed:
            print(f"\n  [Speed] {cfg_name}...")
            try:
                engine = load_engine(cfg["output"])
                ms, std, fps = benchmark_speed_trt(engine, args.n_warmup, args.n_measured, args.batch)
                results["speed"][cfg_name] = {"mean_ms": ms, "std_ms": std, "fps": fps}
                print(f"    {ms:.2f} ± {std:.2f} ms/img, {fps:.1f} FPS")
                del engine
            except Exception as e:
                print(f"    Speed benchmark failed: {e}")
                results["speed"][cfg_name] = {"error": str(e)}

        if do_accuracy:
            print(f"\n  [Accuracy] {cfg_name}...")
            try:
                acc = benchmark_accuracy_trt(cfg["output"], args.data, args.imgsz, args.eval_batch, args.device)
                results["accuracy"][cfg_name] = acc
                print(f"    mAP@0.5={acc['map50']:.4f}, mAP@0.5:0.95={acc['map']:.4f}")
            except Exception as e:
                print(f"    Accuracy benchmark failed: {e}")
                results["accuracy"][cfg_name] = {"error": str(e)}

        if not args.keep_engines and cfg_name != "mixed":
            os.remove(cfg["output"])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
```

### 6.7 python/trt/check_precision_fast.py

```python
"""Fast check script: evaluate polygraphy_debug.engine on 100 COCO images."""
import json, os, sys, time
import cv2, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

DATA_DIR = "/workspace/datasets"
MAX_IMAGES = 100
CONF_THRESH = 0.001
NMS_THRESH = 0.65
THRESHOLD = 0.460

def main():
    engine_path = "polygraphy_debug.engine"
    if not os.path.exists(engine_path):
        print("CHECK: Engine not found")
        sys.exit(1)

    model = YOLO(engine_path)
    ann_file = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")
    coco_gt = COCO(ann_file)
    coco_cats = coco_gt.loadCats(coco_gt.getCatIds())
    ul2coco = {}
    for i, cat in enumerate(sorted(coco_cats, key=lambda x: x["id"])):
        ul2coco[i] = cat["id"]

    img_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    img_infos = coco_gt.loadImgs(img_ids)
    all_results = []

    for idx, img_info in enumerate(img_infos):
        img_path = os.path.join(DATA_DIR, "images", "val2017", img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        results = model.predict(img, imgsz=640, conf=CONF_THRESH, iou=NMS_THRESH, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            h, w = img.shape[:2]
            for i in range(len(boxes)):
                x1 = max(0, float(boxes[i][0]))
                y1 = max(0, float(boxes[i][1]))
                x2 = min(w, float(boxes[i][2]))
                y2 = min(h, float(boxes[i][3]))
                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue
                all_results.append({
                    "image_id": img_info["id"],
                    "category_id": ul2coco[int(cls_ids[i])],
                    "bbox": [x1, y1, bw, bh],
                    "score": float(scores[i])
                })

    if not all_results:
        print("CHECK: No detections, FAIL")
        sys.exit(1)

    coco_dt = coco_gt.loadRes(all_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map50_95 = float(coco_eval.stats[0])

    print("CHECK: mAP@0.5:0.95 = %.4f (threshold=%.4f)" % (map50_95, THRESHOLD))
    if map50_95 >= THRESHOLD:
        print("CHECK: PASS")
        sys.exit(0)
    else:
        print("CHECK: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 6.8 python/trt/check_precision_wrapper.sh

```bash
#!/bin/bash
cd /workspace
python3 python/trt/check_precision_fast.py
```

### 6.9 python/analysis/eval_mixed_precision.py

```python
#!/usr/bin/env python3
"""Mixed Precision INT8+FP16 Accuracy Evaluation for YOLOv11."""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def quantize_weight_per_channel(weight, num_bits=8):
    """Per-channel symmetric INT8 quantization for weights."""
    qmax = 2 ** (num_bits - 1) - 1
    oc = weight.shape[0]
    w_flat = weight.reshape(oc, -1)
    scales = w_flat.abs().max(dim=1).values + 1e-8
    q_w = (w_flat / scales.view(-1, 1) * qmax).round().clamp(-qmax - 1, qmax)
    dq_w = q_w / qmax * scales.view(-1, 1)
    return dq_w.reshape(weight.shape), scales


def quantize_activation_per_tensor(x, num_bits=8):
    """Per-tensor symmetric INT8 quantization for activations."""
    qmax = 2 ** (num_bits - 1) - 1
    scale = x.abs().max() + 1e-8
    q_x = (x / scale * qmax).round().clamp(-qmax - 1, qmax)
    dq_x = q_x / qmax * scale
    return dq_x, scale


class QuantForwardPreHook:
    """Forward pre-hook that quantizes Conv2d weights+inputs for INT8 layers."""

    def __init__(self, int8_layer_names, quantize_input=True):
        self.int8_layer_names = int8_layer_names
        self.quantize_input = quantize_input
        self.original_weights = {}

    def __call__(self, module, inputs):
        layer_name = getattr(module, '_quant_name', None)
        if layer_name is None or layer_name not in self.int8_layer_names:
            return None

        orig_weight = module.weight.data
        q_weight, _ = quantize_weight_per_channel(orig_weight)
        module.weight.data.copy_(q_weight)
        self.original_weights[id(module)] = orig_weight.clone()

        if self.quantize_input:
            x = inputs[0]
            q_x, _ = quantize_activation_per_tensor(x)
            return (q_x,)
        return None


class QuantForwardHook:
    """Hook to restore original weights after forward."""
    def __call__(self, module, inputs, output):
        if id(module) in self.pre_hook.original_weights:
            module.weight.data.copy_(self.pre_hook.original_weights.pop(id(module)))
        return output


def apply_mixed_precision_hooks(model, int8_layer_names, quantize_input=True):
    """Register hooks: INT8 layers quantized, others stay FP32."""
    pre_hook = QuantForwardPreHook(int8_layer_names, quantize_input=quantize_input)
    hook = QuantForwardHook(pre_hook)
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module._quant_name = name
            h_pre = module.register_forward_pre_hook(pre_hook)
            h_post = module.register_forward_hook(hook)
            handles.extend([h_pre, h_post])

    n_int8 = sum(1 for n in int8_layer_names)
    print(f"  INT8 layers: {n_int8}, FP16/FP32 layers: {len(handles)//2 - n_int8}")
    return handles


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Mixed Precision Accuracy Evaluation")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--data", type=str, default="/home/lixiang/work/test/yolo/datasets/coco_local.yaml")
    parser.add_argument("--layer_precision", type=str, default="", help="Layer precision JSON")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output", type=str, default="mixed_precision_results.json")
    parser.add_argument("--no_quant_input", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv11 Mixed Precision (INT8+FP16) Evaluation")
    print("=" * 60)

    with open(args.layer_precision) as f:
        config = json.load(f)

    int8_layer_names = set()
    for detail in config["layer_details"]:
        if detail["precision"] == "INT8":
            int8_layer_names.add(detail["name"])

    print(f"  Precision config: {args.layer_precision}")
    print(f"  INT8 layers: {len(int8_layer_names)}")

    print("\n[1/3] Running FP32 baseline...")
    model = YOLO(args.model)
    results_fp32 = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                              device=args.device, verbose=False, max_det=300)
    fp32_map50 = results_fp32.box.map50
    fp32_map = results_fp32.box.map
    print(f"  FP32: mAP@0.5={fp32_map50:.4f}, mAP@0.5:0.95={fp32_map:.4f}")

    print("\n[2/3] Applying mixed precision INT8+FP16...")
    model_mixed = YOLO(args.model)
    pt_model_mixed = model_mixed.model
    handles = apply_mixed_precision_hooks(
        pt_model_mixed, int8_layer_names, quantize_input=not args.no_quant_input
    )

    print("\n[3/3] Running mixed precision evaluation on COCO val2017...")
    results_mixed = model_mixed.val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                                     device=args.device, verbose=False, max_det=300)

    for h in handles:
        h.remove()

    mixed_map50 = results_mixed.box.map50
    mixed_map = results_mixed.box.map

    drop_map50 = fp32_map50 - mixed_map50
    drop_map = fp32_map - mixed_map
    drop_map50_rel = drop_map50 / fp32_map50 * 100
    drop_map_rel = drop_map / fp32_map * 100

    print(f"\n  Mixed INT8+FP16: mAP@0.5={mixed_map50:.4f}, mAP@0.5:0.95={mixed_map:.4f}")
    print(f"  Drop from FP32: mAP@0.5={drop_map50:.4f} ({drop_map50_rel:.1f}%), mAP={drop_map:.4f} ({drop_map_rel:.1f}%)")

    results = {
        "model": str(Path(args.model).stem),
        "fp32": {"map50": float(fp32_map50), "map": float(fp32_map)},
        "mixed": {"map50": float(mixed_map50), "map": float(mixed_map)},
        "drop": {"map": float(drop_map), "map_percent": float(drop_map_rel)},
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
```

---

## 7. C++ 完整源代码

### 7.1 cpp/CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)

project(YOLOv11TRT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_STANDALONE "Build standalone executable (main.cpp)" ON)
option(BUILD_ROS2      "Build ROS2 perception node"           OFF)

set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
set(TENSORRT_INCLUDE_DIR ${TENSORRT_INCLUDE_DIR} CACHE PATH "TensorRT include path")
set(TENSORRT_LIB_DIR ${TENSORRT_LIB_DIR} CACHE PATH "TensorRT lib path")

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or set OpenCV_DIR.")
endif()

find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please install the CUDA Toolkit.")
endif()

if(WIN32)
    set(TENSORRT_LIBS
        "${TENSORRT_LIB_DIR}/nvinfer.lib"
        "${TENSORRT_LIB_DIR}/nvonnxparser.lib"
        "${TENSORRT_LIB_DIR}/nvparsers.lib"
        "${TENSORRT_LIB_DIR}/nvinfer_plugin.lib"
    )
else()
    set(TENSORRT_LIBS
        nvinfer
        nvonnxparser
        nvinfer_plugin
    )
endif()

set(CORE_SOURCES
    src/yolov11.cpp
    src/preprocess.cu
)

set(CORE_HEADERS
    include/yolov11.h
    include/preprocess.h
    include/common.h
    include/cuda_utils.h
    include/logging.h
    include/macros.h
)

add_library(yolov11trt SHARED ${CORE_SOURCES} ${CORE_HEADERS})

target_compile_definitions(yolov11trt PRIVATE API_EXPORTS)

target_include_directories(yolov11trt PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/yolov11trt>
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(yolov11trt PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)

set_target_properties(yolov11trt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    VERSION 1.0.0
    SOVERSION 1
)

install(TARGETS yolov11trt
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION include/yolov11trt
    FILES_MATCHING PATTERN "*.h"
)

if(BUILD_STANDALONE)
    add_executable(${PROJECT_NAME} main.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE
        yolov11trt
        ${OpenCV_LIBS}
    )
endif()

if(BUILD_ROS2)
    find_package(rclcpp REQUIRED)
    find_package(cv_bridge REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(vision_msgs REQUIRED)

    add_executable(yolov11_ros2_node ros2/perception_node.cpp)
    target_link_libraries(yolov11_ros2_node PRIVATE
        yolov11trt
        rclcpp::rclcpp
        cv_bridge::cv_bridge
        vision_msgs::vision_msgs__rosidl_generator_cpp
        vision_msgs::vision_msgs__rosidl_typesupport_cpp
        sensor_msgs::sensor_msgs__rosidl_generator_cpp
        sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
    )
    target_include_directories(yolov11_ros2_node PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
        /opt/ros/humble/include/vision_msgs
    )

    install(TARGETS yolov11_ros2_node
        DESTINATION lib/${PROJECT_NAME}
    )

    add_executable(yolov11_visualization_node ros2/visualization_node.cpp)
    target_link_libraries(yolov11_visualization_node PRIVATE
        rclcpp::rclcpp
        cv_bridge::cv_bridge
        sensor_msgs::sensor_msgs__rosidl_generator_cpp
        sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
        vision_msgs::vision_msgs__rosidl_generator_cpp
        vision_msgs::vision_msgs__rosidl_typesupport_cpp
    )
    target_include_directories(yolov11_visualization_node PRIVATE
        /opt/ros/humble/include/vision_msgs
    )

    install(TARGETS yolov11_visualization_node
        DESTINATION lib/${PROJECT_NAME}
    )

    install(DIRECTORY ros2/launch/
        DESTINATION share/${PROJECT_NAME}/launch
        FILES_MATCHING PATTERN "*.py"
    )
endif()
```

### 7.2 cpp/include/macros.h

```cpp
#ifndef __MACROS_H
#define __MACROS_H

#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#else

#if defined(_MSC_VER)
#define API __declspec(dllimport)
#else
#define API
#endif
#endif  // API_EXPORTS

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

#endif  // __MACROS_H
```

### 7.3 cpp/include/cuda_utils.h

```cpp
#ifndef TRTX_CUDA_UTILS_H_
#define TRTX_CUDA_UTILS_H_

#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

#endif  // TRTX_CUDA_UTILS_H_
```

### 7.4 cpp/include/common.h

```cpp
const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush" };

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0} };
```

### 7.5 cpp/include/logging.h

```cpp
/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#ifndef TENSORRT_LOGGING_H
#define TENSORRT_LOGGING_H

#include "NvInferRuntimeCommon.h"
#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include "macros.h"

using Severity = nvinfer1::ILogger::Severity;

class LogStreamConsumerBuffer : public std::stringbuf
{
public:
    LogStreamConsumerBuffer(std::ostream& stream, const std::string& prefix, bool shouldLog)
        : mOutput(stream), mPrefix(prefix), mShouldLog(shouldLog) {}
    ~LogStreamConsumerBuffer() {
        if (pbase() != pptr()) { putOutput(); }
    }
    virtual int sync() { putOutput(); return 0; }
    void putOutput() {
        if (mShouldLog) {
            std::time_t timestamp = std::time(nullptr);
            tm* tm_local = std::localtime(&timestamp);
            std::cout << "[" << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/"
                      << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/"
                      << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-"
                      << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":"
                      << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":"
                      << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
            mOutput << mPrefix << str();
            str("");
            mOutput.flush();
        }
    }
    void setShouldLog(bool shouldLog) { mShouldLog = shouldLog; }
private:
    std::ostream& mOutput;
    std::string mPrefix;
    bool mShouldLog;
};

class LogStreamConsumerBase {
public:
    LogStreamConsumerBase(std::ostream& stream, const std::string& prefix, bool shouldLog)
        : mBuffer(stream, prefix, shouldLog) {}
protected:
    LogStreamConsumerBuffer mBuffer;
};

class LogStreamConsumer : protected LogStreamConsumerBase, public std::ostream {
public:
    LogStreamConsumer(Severity reportableSeverity, Severity severity)
        : LogStreamConsumerBase(severityOstream(severity), severityPrefix(severity), severity <= reportableSeverity)
        , std::ostream(&mBuffer)
        , mShouldLog(severity <= reportableSeverity)
        , mSeverity(severity) {}
    void setReportableSeverity(Severity severity) {
        mShouldLog = mSeverity <= severity;
        mBuffer.setShouldLog(mShouldLog);
    }
private:
    static std::ostream& severityOstream(Severity severity) {
        return severity >= Severity::kINFO ? std::cout : std::cerr;
    }
    static std::string severityPrefix(Severity severity) {
        switch (severity) {
        case Severity::kINTERNAL_ERROR: return "[F] ";
        case Severity::kERROR: return "[E] ";
        case Severity::kWARNING: return "[W] ";
        case Severity::kINFO: return "[I] ";
        case Severity::kVERBOSE: return "[V] ";
        default: assert(0); return "";
        }
    }
    bool mShouldLog;
    Severity mSeverity;
};

class Logger : public nvinfer1::ILogger {
public:
    Logger(Severity severity = Severity::kWARNING) : mReportableSeverity(severity) {}
    void log(Severity severity, const char* msg) TRT_NOEXCEPT override {
        LogStreamConsumer(mReportableSeverity, severity) << "[TRT] " << std::string(msg) << std::endl;
    }
    void setReportableSeverity(Severity severity) { mReportableSeverity = severity; }
    Severity getReportableSeverity() const { return mReportableSeverity; }
private:
    Severity mReportableSeverity;
};

#endif // TENSORRT_LOGGING_H
```

### 7.6 cpp/include/preprocess.h

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

void cuda_preprocess_init(int max_image_size);
void cuda_preprocess_destroy();
void cuda_preprocess(uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream);
```

### 7.7 cpp/include/yolov11.h

```cpp
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
```

### 7.8 cpp/src/preprocess.cu

```cuda
#include "preprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"

static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;

struct AffineMatrix {
    float value[6];
};

__global__ void warpaffine_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    uint8_t const_value_st, AffineMatrix d2s, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;

    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;

    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } else {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;

        float w1 = hy * hx;
        float w2 = hy * lx;
        float w3 = ly * hx;
        float w4 = ly * lx;

        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;
            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;
            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    float t = c2;
    c2 = c0;
    c0 = t;

    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;

    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void cuda_preprocess(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream)
{
    int img_size = src_width * src_height * 3;
    memcpy(img_buffer_host, src, img_size);

    CUDA_CHECK(cudaMemcpyAsync(
        img_buffer_device, img_buffer_host, img_size,
        cudaMemcpyHostToDevice, stream));

    AffineMatrix s2d, d2s;

    float scale = std::min(
        dst_height / (float)src_height,
        dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width * 0.5f + dst_width * 0.5f;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5f + dst_height * 0.5f;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);

    warpaffine_kernel << <blocks, threads, 0, stream >> > (
        img_buffer_device, src_width * 3, src_width, src_height,
        dst, dst_width, dst_height, 128, d2s, jobs);

    CUDA_CHECK(cudaGetLastError());
}

void cuda_preprocess_init(int max_image_size) {
    CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));
    CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

void cuda_preprocess_destroy() {
    CUDA_CHECK(cudaFree(img_buffer_device));
    CUDA_CHECK(cudaFreeHost(img_buffer_host));
}
```

### 7.9 cpp/src/yolov11.cpp

```cpp
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
```

### 7.10 cpp/main.cpp

```cpp
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
    std::cout << "  image_path:  Path to input image" << std::endl;
    std::cout << "  engine_path: Path to TensorRT engine file" << std::endl;
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
```

### 7.11 cpp/ros2/perception_node.cpp

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <vision_msgs/msg/bounding_box2_d.hpp>
#include <cv_bridge/cv_bridge.h>

#include "yolov11.h"

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
        this->declare_parameter<bool>("publish_debug_image", false);

        auto engine_path = this->get_parameter("engine_path").as_string();
        auto image_topic = this->get_parameter("image_topic").as_string();
        auto conf_thresh = this->get_parameter("conf_threshold").as_double();
        auto nms_thresh = this->get_parameter("nms_threshold").as_double();
        auto pub_debug_img = this->get_parameter("publish_debug_image").as_bool();

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
        if (pub_debug_img) {
            debug_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>("detection_image", 10);
        }

        img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic, rclcpp::QoS(10).best_effort(),
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

        std::vector<Detection> detections;
        try {
            yolo_->preprocess(image);
            yolo_->infer();
            yolo_->postprocess(detections);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Inference error: %s", e.what());
            return;
        }

        publishDetections(msg->header, detections);

        if (debug_img_pub_ && this->count_subscribers(debug_img_pub_->get_topic_name()) > 0) {
            cv::Mat display = image.clone();
            yolo_->draw(display, detections);
            auto debug_msg = cv_bridge::CvImage(msg->header, "bgr8", display).toImageMsg();
            debug_img_pub_->publish(*debug_msg);
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
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_img_pub_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<YOLOv11PerceptionNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
```

### 7.12 cpp/ros2/visualization_node.cpp

```cpp
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
```

### 7.13 cpp/ros2/launch/perception_with_viz.launch.py

```python
"""Launch YOLOv11 perception node + visualization + image_view."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

PERCEPTION_NODE = '/usr/local/lib/YOLOv11TRT/yolov11_ros2_node'
VIZ_NODE = '/usr/local/lib/YOLOv11TRT/yolov11_visualization_node'


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('engine_path', description='Path to TensorRT engine file'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('detections_topic', default_value='/detections'),
        DeclareLaunchArgument('output_topic', default_value='/detection_image'),
        DeclareLaunchArgument('conf_threshold', default_value='0.3'),
        DeclareLaunchArgument('nms_threshold', default_value='0.4'),

        Node(
            executable=PERCEPTION_NODE,
            name='yolov11_perception',
            output='screen',
            parameters=[{
                'engine_path': LaunchConfiguration('engine_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'nms_threshold': LaunchConfiguration('nms_threshold'),
                'publish_debug_image': False,
            }],
            remappings=[('detections', LaunchConfiguration('detections_topic'))],
        ),

        Node(
            executable=VIZ_NODE,
            name='yolov11_visualization',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'detections_topic': LaunchConfiguration('detections_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
            }],
        ),

        Node(
            package='image_view',
            executable='image_view',
            name='detection_viewer',
            output='screen',
            remappings=[('image', LaunchConfiguration('output_topic'))],
        ),
    ])
```

### 7.14 cpp/scrip/test_image_pub.py

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.5, self.publish_callback)

        possible_paths = [
            '/workspace/cpp/1.png',
            '/home/lixiang/work/test/yolo/ros_yolov11_trt/cpp/1.png',
            os.path.join(os.path.dirname(__file__), '1.png'),
        ]

        self.image = None
        for path in possible_paths:
            if os.path.exists(path):
                self.image = cv2.imread(path)
                self.get_logger().info(f'Loaded image from: {path}')
                break

        if self.image is None:
            self.get_logger().error('Image not found, tried: ' + str(possible_paths))
        else:
            self.get_logger().info(f'Image shape: {self.image.shape}')

    def publish_callback(self):
        if self.image is not None:
            msg = self.bridge.cv2_to_imgmsg(self.image, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera"
            self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclcpp.shutdown()

if __name__ == '__main__':
    main()
```

---

## 8. configs/layer_precision.json

此文件由敏感性分析自动生成，示例结构如下：

```json
{
  "model": "yolo11n",
  "sensitivity_analysis_method": "per_layer_cosine",
  "baseline_score": 0.9996965704485774,
  "sensitivity_threshold": 0.05,
  "baseline_mAP": 0.0,
  "total_layers": 88,
  "sensitive_layers_count": 25,
  "default_precision": "INT8",
  "sensitive_layers": [
    "model.23.cv3.0.1.0.conv",
    "model.23.cv3.1.2",
    "model.23.cv3.2.1.1.conv",
    "model.23.cv2.0.2",
    "model.23.cv2.1.0.conv",
    "model.23.cv2.2.2",
    "model.23.cv2.0.1.conv",
    "model.23.cv2.2.0.conv",
    "model.23.cv3.2.2",
    "model.23.cv3.0.0.0.conv",
    "model.23.cv3.0.2",
    "model.23.cv2.1.2",
    "model.23.cv3.1.1.0.conv",
    "model.23.cv2.2.1.conv",
    "model.23.cv2.1.1.conv",
    "model.23.cv3.1.1.1.conv",
    "model.23.cv3.2.0.1.conv",
    "model.23.cv3.2.0.0.conv",
    "model.23.cv3.1.0.0.conv",
    "model.23.cv3.1.0.1.conv",
    "model.23.cv3.2.1.0.conv",
    "model.23.cv2.0.0.conv",
    "model.23.dfl.conv",
    "model.23.cv3.0.1.1.conv",
    "model.23.cv3.0.0.1.conv"
  ],
  "sensitive_precision": "FP16",
  "layer_details": [
    {"name": "model.0.conv", "score": 1.00467, "precision": "FP16"},
    {"name": "model.1.conv", "score": 1.000399, "precision": "FP16"}
  ],
  "config_name": "backbone_fp16_head_neck_int8",
  "notes": "Optimal config: backbone FP16 preserves accuracy (-0.34% drop), head/neck INT8."
}
```

---

## 9. 编译与运行

### 9.1 C++ 独立版本编译

```bash
cd cpp
mkdir -p build && cd build
cmake .. -DTENSORRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
         -DTENSORRT_LIB_DIR=/usr/lib/x86_64-linux-gnu
make -j$(nproc)
```

### 9.2 ROS2 版本编译

```bash
cd cpp
mkdir -p build_ros2 && cd build_ros2
source /opt/ros/humble/setup.bash
cmake .. -DBUILD_STANDALONE=OFF -DBUILD_ROS2=ON
make -j$(nproc)
make install
```

### 9.3 运行 ROS2 节点

```bash
# 终端1：启动感知+可视化
source /opt/ros/humble/setup.bash
ros2 launch /usr/local/share/YOLOv11TRT/launch/perception_with_viz.launch.py \
    engine_path:=../../weights/yolo11n_forward_mixed.engine

# 终端2：发布测试图片
source /opt/ros/humble/setup.bash
python3 /workspace/cpp/scrip/test_image_pub.py
```

### 9.4 独立推理测试

```bash
cd cpp/build
./YOLOv11TRT /path/to/test.jpg /path/to/engine.engine
```

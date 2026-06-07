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
            print(f"  Processing batch {batch_idx}... ({samples_processed}/{num_samples})")

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
    print(f"  Found {len(pt_model.model)} sequential modules")

    print("\n[2/4] Collecting Conv2d layers...")
    conv_layers = collect_conv_layers(pt_model)
    print(f"  Found {len(conv_layers)} Conv2d layers")

    categories = detect_head_layer_names(conv_layers)
    for cat, layers in categories.items():
        print(f"    {cat}: {len(layers)} layers")

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

    print(f"\n  Sensitivity analysis complete!")
    print(f"  Top-10 most sensitive layers:")
    for name, score in sensitivity_results[:10]:
        print(f"    {name:50s} score={score:.6f}")

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
    print(f"  INT8 layers: {config['total_layers'] - config['sensitive_layers_count']}")
    print(f"  FP16 layers: {config['sensitive_layers_count']}")


if __name__ == "__main__":
    main()
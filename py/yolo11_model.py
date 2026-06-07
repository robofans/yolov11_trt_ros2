#!/usr/bin/env python3
"""YOLOv11 model definitions for analysis."""

from ultralytics import YOLO
import torch
import torch.nn as nn


def get_yolo11_model(model_path="yolo11n.pt"):
    """Load YOLOv11 model from path."""
    model = YOLO(model_path)
    return model


def get_conv_layers(model):
    """Get all Conv2d layers from YOLO model."""
    conv_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers[name] = module
    return conv_layers


def get_model_info(model):
    """Get model layer information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_count = sum(1 for _ in model.modules() if isinstance(_, nn.Conv2d))

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "conv_layers": conv_count,
    }


if __name__ == "__main__":
    model = get_yolo11_model()
    info = get_model_info(model.model)
    print(f"Model info: {info}")
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
    print(f"         imgsz={args.imgsz}, batch={args.batch}, opset={args.opset}")

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
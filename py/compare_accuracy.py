#!/usr/bin/env python3
"""Speed + Accuracy comparison: one val pass per model, batch=1."""

# fmt: off
MODELS = [
    # ("pytorch",      "weights/yolo11n.pt"),
    # ("trt_fp32",     "weights/yolo11n_fp32.engine"),
    # ("trt_fp16",     "weights/yolo11n_fp16.engine"),
    ("trt_int8",     "weights/yolo11n_int8.engine"),
    ("trt_mixed",    "weights/yolo11n_head3_tail32_mixed.engine"),
]
# fmt: on

import argparse
import json
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Speed + Accuracy comparison (one val pass per model)")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output", type=str, default="results/compare_results.json")
    args = parser.parse_args()

    results = {}
    baseline = None

    for name, path in MODELS:
        if not os.path.exists(path):
            print(f"\n  SKIP {name}: not found ({path})")
            results[name] = {"error": "not found"}
            continue

        print(f"\n{'=' * 50}")
        print(f"  [{name}]  {path}")
        print(f"{'=' * 50}")

        try:
            model = YOLO(path)
            r = model.val(data=args.data, imgsz=args.imgsz, batch=1,
                          device=args.device, verbose=False, max_det=300)
            inf_ms = float(r.speed.get("inference", 0.0)) if r.speed else 0.0
            entry = {
                "map50": float(r.box.map50),
                "map": float(r.box.map),
                "mean_ms": inf_ms,
                "fps": 1000.0 / inf_ms if inf_ms > 0 else 0.0,
            }
            results[name] = entry
            if baseline is None:
                baseline = entry
            print(f"    mAP@0.5={entry['map50']:.4f}  mAP={entry['map']:.4f}  "
                  f"{entry['fps']:.1f} FPS  ({inf_ms:.2f} ms/img)")
        except Exception as e:
            print(f"    FAIL: {e}")
            results[name] = {"error": str(e)}

    # Print table
    print(f"\n{'=' * 72}")
    print(f"  {'Model':<16} {'FPS':>8} {'ms':>8} {'mAP@0.5':>8} {'mAP':>8} {'ΔmAP@0.5':>10} {'ΔmAP':>8}")
    print(f"  {'-' * 72}")

    for name, _ in MODELS:
        m = results.get(name, {})
        if "error" in m:
            print(f"  {name:<16} {'SKIPPED':>8}")
            continue
        d50 = m["map50"] - baseline["map50"] if baseline else 0
        dm = m["map"] - baseline["map"] if baseline else 0
        print(f"  {name:<16} {m['fps']:8.1f} {m['mean_ms']:8.2f} "
              f"{m['map50']:8.4f} {m['map']:8.4f} {d50:+10.4f} {dm:+8.4f}")

    print(f"{'=' * 72}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({"models": [n for n, _ in MODELS], "baseline": baseline,
                   "results": results}, f, indent=2)
    print(f"\n  Saved to: {args.output}")


if __name__ == "__main__":
    main()

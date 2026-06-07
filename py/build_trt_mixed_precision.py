#!/usr/bin/env python3
"""
Build mixed-precision TensorRT engine with head/tail FP16 protection.

All Conv layers default to INT8, except:
  --protect-head N   → first N Conv layers run FP16
  --protect-tail M   → last M Conv layers run FP16

Example:
  # Protect last 32 Conv layers (detection head + DFG) with FP16
  python build_mixed_precision.py weights/yolo11n.onnx --protect-tail 32 --calib-dir calib_data

  # Protect first 3 and last 32 Conv layers
  python build_mixed_precision.py weights/yolo11n.onnx --protect-head 3 --protect-tail 32 --calib-dir calib_data
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_trt_norm_engine import Calibrator

os.environ["TRT_CASK_DISABLE"] = "1"
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def collect_conv_indices(network):
    """Return list of (layer_index, name) for all Convolution layers in network order."""
    result = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type == trt.LayerType.CONVOLUTION:
            result.append((i, layer.name))
    return result


def apply_head_tail_fp16(network, n_head, n_tail):
    """Mark first n_head and last n_tail Conv layers as FP16.

    Returns (fp16_count, int8_count, head_names, tail_names).
    """
    conv_layers = collect_conv_indices(network)
    n_total = len(conv_layers)

    if n_total == 0:
        print("  Warning: no Conv layers found in network")
        return 0, 0, [], []

    head_indices = set()
    tail_indices = set()

    if n_head > 0:
        head_count = min(n_head, n_total)
        head_indices = {conv_layers[j][0] for j in range(head_count)}
        names = [conv_layers[j][1] for j in range(head_count)]
        print(f"  Head protected: first {head_count} Conv layers → FP16")
        for n in names[:5]:
            print(f"    {n}")
        if len(names) > 5:
            print(f"    ... and {len(names) - 5} more")

    if n_tail > 0:
        tail_count = min(n_tail, n_total)
        tail_indices = {conv_layers[j][0] for j in range(n_total - tail_count, n_total)}
        names = [conv_layers[j][1] for j in range(n_total - tail_count, n_total)]
        print(f"  Tail protected: last {tail_count} Conv layers → FP16")
        for n in names[:5]:
            print(f"    {n}")
        if len(names) > 5:
            print(f"    ... and {len(names) - 5} more")

    protected = head_indices | tail_indices
    fp16_count = 0
    int8_count = 0

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type != trt.LayerType.CONVOLUTION:
            continue

        if i in protected:
            layer.precision = trt.DataType.HALF
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.DataType.HALF)
            fp16_count += 1
        # non-protected layers are left at default → TRT quantizes to INT8

    int8_count = n_total - fp16_count
    print(f"  Result: {fp16_count} FP16 + {int8_count} INT8")
    return fp16_count, int8_count, list(head_indices), list(tail_indices)


def build_engine(onnx_path, output_path, n_head=0, n_tail=0,
                 calib_dir=None, workspace_size=1 << 30, batch_size=1):
    """Build mixed-precision TensorRT engine."""
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
                print(f"  Parse error {err}: {parser.get_error(err)}")
            raise RuntimeError("ONNX parse failed")

    print(f"  Network: {network.num_layers} layers")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    # Enable FP16 and INT8, let layer precision constraints decide per layer
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    print("[Build] Applying head/tail FP16 protection...")
    apply_head_tail_fp16(network, n_head, n_tail)

    if calib_dir and os.path.isdir(calib_dir):
        calibrator = Calibrator(calib_dir, batch_size=32)
        config.int8_calibrator = calibrator
        print(f"  Calibrator initialized: {calib_dir}")

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
    print(f"  Output: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build mixed-precision TRT engine (head/tail FP16, rest INT8)"
    )
    parser.add_argument("onnx", type=str, help="Input ONNX model path")
    parser.add_argument("--output", "-o", type=str, default="", help="Output .engine path")
    parser.add_argument("--protect-head", type=int, default=0,
                        help="First N Conv layers to run in FP16")
    parser.add_argument("--protect-tail", type=int, default=0,
                        help="Last M Conv layers to run in FP16")
    parser.add_argument("--calib-dir", type=str, default="calib_data",
                        help="INT8 calibration data directory")
    parser.add_argument("--workspace", type=int, default=1 << 30,
                        help="Workspace size in bytes")
    parser.add_argument("--batch", type=int, default=1, help="Max batch size")
    args = parser.parse_args()

    if args.protect_head == 0 and args.protect_tail == 0:
        parser.error("At least one of --protect-head or --protect-tail must be > 0")

    if not args.output:
        stem = Path(args.onnx).stem
        parts = [stem]
        if args.protect_head > 0:
            parts.append(f"head{args.protect_head}")
        if args.protect_tail > 0:
            parts.append(f"tail{args.protect_tail}")
        parts.append("mixed")
        args.output = str(Path(args.onnx).parent / f"{'_'.join(parts)}.engine")

    build_engine(
        onnx_path=args.onnx,
        output_path=args.output,
        n_head=args.protect_head,
        n_tail=args.protect_tail,
        calib_dir=args.calib_dir,
        workspace_size=args.workspace,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()

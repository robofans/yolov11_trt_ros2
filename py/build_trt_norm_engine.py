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

    print(f"  Layer precision applied: INT8={int8_count}, FP16={fp16_count}, FP32={fp32_count}")
    unmatched = set(layer_map.keys()) - matched_names
    if unmatched:
        print(f"  Unmatched layer_map entries: {len(unmatched)}")
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
                print(f"  Parse error {err}: {parser.get_error(err)}")
            raise RuntimeError("ONNX parse failed")

    print(f"  Network: {network.num_layers} layers")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

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
            print(f"  Calibrator initialized: {calib_dir}")
        if layer_precision:
            print(f"[Build] Loading layer precision: {layer_precision}")
            layer_map, default_prec, sensitive = load_layer_precision(layer_precision)
            print(f"  Default precision: {'INT8' if default_prec == trt.DataType.INT8 else 'FP16'}")
            print(f"  Sensitive layers: {len(sensitive)} (FP16)")
            apply_layer_precision(network, layer_map, default_prec)
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

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
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("onnx", type=str, help="Input ONNX model path")
    parser.add_argument("--output", "-o", type=str, default="", help="Output .engine path")
    prec_group = parser.add_mutually_exclusive_group()
    prec_group.add_argument("--fp32", action="store_true", help="Enable FP32 (default)")
    prec_group.add_argument("--fp16", action="store_true", help="Enable FP16")
    prec_group.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--calib_dir", type=str, default="", help="Calibration data directory")
    parser.add_argument("--layer_precision", type=str, default="", help="Layer precision JSON")
    parser.add_argument("--workspace", type=int, default=1 << 30, help="Workspace size (bytes)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    if not args.output:
        stem = Path(args.onnx).stem
        if args.int8 and args.layer_precision:
            prec = "mixed"
        elif args.int8:
            prec = "int8"
        elif args.fp16:
            prec = "fp16"
        else:
            prec = "fp32"
        args.output = str(Path(args.onnx).parent / f"{stem}_{prec}.engine")

    build_engine(
        onnx_path=args.onnx, output_path=args.output,
        fp16=args.fp16, int8=args.int8,
        calib_dir=args.calib_dir, layer_precision=args.layer_precision,
        workspace_size=args.workspace, batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
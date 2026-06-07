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
        print(f"  Warning: only {len(sampled)} images available (requested {num_samples})")
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
    print(f"  Images: {num_images}")
    print(f"  Size:   {input_w}x{input_h}")
    print(f"  Output: {bin_path}")

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
                print(f"  Warning: Failed to process {img_path}: {e}")
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
    print(f"  Found {len(image_paths)} images")

    print("\n[2/2] Preprocessing and saving calibration data...")
    prepare_calibration_data(image_paths, args.output_dir, input_w=args.input_w,
                             input_h=args.input_h, batch_size=args.batch_size)

    print("\n Calibration data ready!")


if __name__ == "__main__":
    main()
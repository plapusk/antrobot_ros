#!/usr/bin/env python3
# Copyright (C) 2025 Dan Novischi. All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v3 or any later version.

"""Standalone inference-speed benchmark for candidate perception models.

Not a ROS node and not part of the package build — run directly with
python3. Uses synthetic frames so it has no camera/ROS dependency.
"""

import argparse
import os
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

MODEL_FILES = {
    'yolo': 'slow_color.pt',
    'deeplab': 'fine_tune_myset.pth',
    'fastscnn': 'zfinal_SCNN_aug_Dice_Adamw.pth',
}


def load_yolo(model_dir):
    path = os.path.join(model_dir, MODEL_FILES['yolo'])
    if not os.path.isfile(path):
        return None, f"model file not found: {path}"
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        return None, f"ultralytics not installed: {exc}"
    try:
        model = YOLO(path)
    except Exception as exc:
        return None, f"failed to load: {exc}"
    return model, None


def load_deeplab(model_dir):
    path = os.path.join(model_dir, MODEL_FILES['deeplab'])
    if not os.path.isfile(path):
        return None, f"model file not found: {path}"
    try:
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet50
    except ImportError as exc:
        return None, f"torch/torchvision not installed: {exc}"
    try:
        model = deeplabv3_resnet50(weights=None, num_classes=4)
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as exc:
        return None, f"failed to load: {exc}"
    return model, None


def load_fastscnn(model_dir):
    return None, "Fast-SCNN architecture not implemented — skipping"


LOADERS = {
    'yolo': load_yolo,
    'deeplab': load_deeplab,
    'fastscnn': load_fastscnn,
}


def predict_yolo(model, frame):
    model.predict(frame, verbose=False)


def predict_deeplab(model, frame):
    import torch
    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        model(tensor)


PREDICT_FNS = {
    'yolo': predict_yolo,
    'deeplab': predict_deeplab,
}


def benchmark_model(model, predict_fn, resolution, iters):
    w, h = resolution
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # warmup
    predict_fn(model, frame)

    start = time.time()
    for _ in range(iters):
        predict_fn(model, frame)
    elapsed = time.time() - start

    avg_latency = elapsed / iters
    hz = 1.0 / avg_latency if avg_latency > 0 else float('inf')
    return avg_latency, hz


def parse_resolutions(arg):
    resolutions = []
    for part in arg.split(','):
        w, h = part.lower().split('x')
        resolutions.append((int(w), int(h)))
    return resolutions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default='yolo,deeplab,fastscnn',
                         help='Comma list of: yolo, deeplab, fastscnn')
    parser.add_argument('--resolutions', default='320x240,640x480',
                         help='Comma list of WxH, e.g. 320x240,640x480')
    parser.add_argument('--iters', type=int, default=5,
                         help='Timed iterations per (model, resolution) combo')
    parser.add_argument('--threads', type=int, default=2,
                         help='torch.set_num_threads value')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                         help='Directory containing model weight files')
    args = parser.parse_args()

    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        pass

    model_names = [m.strip() for m in args.models.split(',') if m.strip()]
    resolutions = parse_resolutions(args.resolutions)

    rows = []
    for name in model_names:
        if name not in LOADERS:
            print(f"Unknown model '{name}', skipping")
            continue

        model, error = LOADERS[name](args.model_dir)
        if model is None:
            rows.append((name, '-', f"SKIPPED ({error})"))
            continue

        predict_fn = PREDICT_FNS[name]
        for resolution in resolutions:
            res_str = f"{resolution[0]}x{resolution[1]}"
            try:
                avg_latency, hz = benchmark_model(model, predict_fn, resolution, args.iters)
                rows.append((name, res_str, f"{avg_latency * 1000:.1f} ms  {hz:.2f} Hz"))
            except Exception as exc:
                rows.append((name, res_str, f"ERROR ({exc})"))

    print(f"{'Model':<10} {'Resolution':<12} {'Result'}")
    for name, res_str, result in rows:
        print(f"{name:<10} {res_str:<12} {result}")


if __name__ == '__main__':
    main()

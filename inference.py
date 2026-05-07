import sys
import os
import json
import cv2
import functools
import argparse
import collections.abc
from types import ModuleType, SimpleNamespace
from tqdm import tqdm
import numpy as np
import torch
from easydict import EasyDict as edict

# --- 1. Compatibility Patches (for PyTorch 2.0+) ---
if 'torch._six' not in sys.modules:
    _six = ModuleType("torch._six")
    _six.container_abcs = collections.abc
    _six.string_classes = (str,)
    _six.int_classes = (int,)
    _six.inf = float('inf')
    sys.modules["torch._six"] = _six

try:
    from torch.serialization import add_safe_globals
    add_safe_globals([argparse.Namespace, np.core.multiarray.scalar, np._core.multiarray.scalar])
except:
    pass

if not hasattr(torch, '__patched_for_aic'):
    _orig_load = torch.load
    @functools.wraps(_orig_load)
    def _patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
    torch.__patched_for_aic = True

# --- 2. Mock optional dependencies to prevent import errors ---
for name, attrs in [
    ("visdom", {"Visdom": type("Visdom", (), {"__init__": lambda s, *a, **k: None})}),
    ("visdom.server", {}),
    ("jpeg4py", {"JPEG": type("JPEG", (), {"__init__": lambda s, *a, **k: None, "decode": lambda s: None})}),
    ("tikzplotlib", {}),
    ("wandb", {}),
    ("lmdb", {}),
    ("pycocotools", {}),
    ("pycocotools.coco", {}),
]:
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)

# --- 3. Dataset Class ---
class AICDataset:
    def __init__(self, root, manifest_path, split='public_lb'):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        if split not in manifest:
            raise ValueError(f"Split '{split}' not found. Available: {list(manifest.keys())}")

        self.sequences = []
        for seq_key, info in manifest[split].items():
            video_path = os.path.join(root, info['video_path'])
            anno_path = os.path.join(root, info['annotation_path'])

            if not os.path.exists(video_path):
                print(f"Warning: Missing video: {video_path}")
                continue
            if not os.path.exists(anno_path):
                print(f"Warning: Missing anno: {anno_path}")
                continue

            with open(anno_path, 'r') as af:
                line = af.readline().strip()
            parts = [float(x) for x in line.replace('\t', ' ').replace(',', ' ').split() if x]

            self.sequences.append({
                'key': seq_key,
                'video': video_path,
                'n_frames': info['n_frames'],
                'gt_bbox': parts[:4],
            })

        self.sequences.sort(key=lambda x: x['key'])
        total_frames = sum(s['n_frames'] for s in self.sequences)
        print(f"Dataset Loaded: Split='{split}' | {len(self.sequences)} sequences | {total_frames:,} frames")

    def __len__(self):
        return len(self.sequences)

# --- 4. Main Inference Function ---
def run_inference(args):
    # Setup Paths
    sys.path.insert(0, os.path.abspath("src"))
    
    # Patch env_settings BEFORE imports
    import lib.test.evaluation.environment as env_mod
    def patched_env_settings():
        s = SimpleNamespace()
        s.prj_dir = os.path.abspath(".")
        s.save_dir = os.path.abspath("output")
        return s
    env_mod.env_settings = patched_env_settings

    # Imports from src
    from lib.test.tracker.sglatrack import sglatrack as SGLATracker
    from lib.test.parameter.sglatrack import parameters

    # Load Model
    params = parameters(args.config)
    params.checkpoint = args.weights
    params.debug = False
    params.save_all_boxes = False

    tracker = SGLATracker(params, 'uav123')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tracker.network.to(device)
    print(f"Model loaded on {device}")

    # Prepare Dataset
    dataset = AICDataset(args.data_root, args.manifest, args.split)

    # Generate Submission
    results = ["id,x,y,w,h\n"]
    for seq in tqdm(dataset.sequences, desc="Tracking"):
        key, gt, n_frames = seq['key'], seq['gt_bbox'], seq['n_frames']
        cap = cv2.VideoCapture(seq['video'])
        if not cap.isOpened():
            print(f"Error: Cannot open: {seq['video']}")
            continue

        initialized = False
        pred = gt

        for idx in range(n_frames):
            ret, frame_bgr = cap.read()
            if not ret:
                results.append(f"{key}_{idx},{pred[0]:.3f},{pred[1]:.3f},{pred[2]:.3f},{pred[3]:.3f}\n")
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            H, W = frame_rgb.shape[:2]

            if idx == 0:
                x = max(0.0, min(gt[0], W - 1))
                y = max(0.0, min(gt[1], H - 1))
                w = max(10.0, min(gt[2], W - x))
                h = max(10.0, min(gt[3], H - y))
                x = min(x, W - w)
                y = min(y, H - h)
                try:
                    tracker.initialize(frame_rgb, {'init_bbox': [x, y, w, h]})
                    pred = [x, y, w, h]
                    initialized = True
                except Exception as e:
                    print(f"Init failed for {key}: {e}")
                    pred = [x, y, w, h]
            else:
                if initialized:
                    try:
                        pred = tracker.track(frame_rgb)['target_bbox']
                    except Exception as e:
                        print(f"Tracking failed for {key} at frame {idx}: {e}")

            results.append(f"{key}_{idx},{pred[0]:.3f},{pred[1]:.3f},{pred[2]:.3f},{pred[3]:.3f}\n")

        cap.release()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.writelines(results)
    print(f"Submission saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLATrack Competition Inference")
    parser.add_argument("--config", type=str, default="deit_distilled", help="Model config name")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint .pth.tar")
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--manifest", type=str, required=True, help="Path to contestant_manifest.json")
    parser.add_argument("--split", type=str, default="public_lb", help="Manifest split (public_lb or private_lb)")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    
    args = parser.parse_args()
    run_inference(args)

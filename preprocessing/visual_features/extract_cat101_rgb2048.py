import os
import argparse, csv, re, json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


CAT_ROOT = Path(os.environ.get("CATARACT101_ROOT", "/path/to/cataract-101"))
DATA_ROOT = Path("/path/to/MS-TCN2/data/cataract101_m10_a0_rgb2048")
OUT_DIR = DATA_ROOT / "features"
META_DIR = DATA_ROOT / "meta"


def get_first_annotation_frames():
    first = {}
    with open(CAT_ROOT / "annotations.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            vid = int(r["VideoID"])
            fr = int(r["FrameNo"])
            first[vid] = min(first.get(vid, fr), fr)
    return first


def get_video_ids_from_bundle(bundle_name="all.bundle"):
    bundle = DATA_ROOT / "splits" / bundle_name
    ids = []
    with open(bundle, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            m = re.match(r"case_(\d+)\.txt", name)
            if m:
                ids.append(int(m.group(1)))
    return ids


def build_model(device):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model


def preprocess_batch(frames):
    arr = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        arr.append(frame)

    x = np.stack(arr).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(0, 3, 1, 2)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def count_gt_lines(vid):
    gt = DATA_ROOT / "groundTruth" / f"case_{vid}.txt"
    with open(gt, encoding="utf-8") as f:
        return sum(1 for _ in f)


def extract_one(vid, model, device, batch_size=64, overwrite=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / f"case_{vid}.npy"
    meta_path = META_DIR / f"case_{vid}_rgb2048.json"

    gt_len = count_gt_lines(vid)

    if out_path.exists() and not overwrite:
        arr = np.load(out_path, mmap_mode="r")
        if arr.shape == (2048, gt_len):
            print(f"[SKIP] case_{vid}: exists {arr.shape}")
            return
        else:
            print(f"[WARN] existing shape mismatch, redo case_{vid}: {arr.shape} vs (2048,{gt_len})")

    first_map = get_first_annotation_frames()
    start_frame = first_map[vid]

    video_path = CAT_ROOT / "videos" / f"case_{vid}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    feats = []
    frames = []
    used = 0
    idx = 0

    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx >= start_frame and used < gt_len:
                frames.append(frame)
                used += 1

                if len(frames) == batch_size:
                    x = preprocess_batch(frames).to(device, non_blocking=True)
                    y = model(x).detach().float().cpu().numpy()
                    feats.append(y)
                    frames = []

            idx += 1

            if used >= gt_len:
                break

        if frames:
            x = preprocess_batch(frames).to(device, non_blocking=True)
            y = model(x).detach().float().cpu().numpy()
            feats.append(y)

    cap.release()

    if used != gt_len:
        raise RuntimeError(f"case_{vid}: used frames {used} != gt_len {gt_len}")

    feat = np.concatenate(feats, axis=0).astype(np.float32)  # T x 2048
    feat = feat.T  # 2048 x T

    if feat.shape != (2048, gt_len):
        raise RuntimeError(f"case_{vid}: feature shape bad {feat.shape}, expected (2048,{gt_len})")

    np.save(out_path, feat)

    info = {
        "video_id": vid,
        "video_path": str(video_path),
        "out_path": str(out_path),
        "start_frame": start_frame,
        "gt_len": gt_len,
        "feature_shape": list(feat.shape),
        "dtype": "float32",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[OK] case_{vid}: {feat.shape} saved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="all.bundle")
    ap.add_argument("--video_id", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = build_model(device)

    if args.video_id is not None:
        vids = [args.video_id]
    else:
        vids = get_video_ids_from_bundle(args.bundle)

    print("num videos:", len(vids))

    for i, vid in enumerate(vids, 1):
        print(f"\n===== [{i}/{len(vids)}] case_{vid} =====")
        extract_one(
            vid=vid,
            model=model,
            device=device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

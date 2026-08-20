import os
import argparse
import json
import shutil
from pathlib import Path

import cv2
import timm
import numpy as np
import torch


ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))
CAT = Path(os.environ.get("CATARACT101_ROOT", "/path/to/cataract-101"))

A0 = ROOT / "data/cataract101_m10_a0_rgb2048"
YOLO = ROOT / "data/cataract101_m10_yolo_b2_multiinst_assets"
OUT = ROOT / "data/cataract101_m10_a2b_x5_nfnet"


def prepare_out():
    for sub in ["features", "groundTruth", "splits", "meta"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    shutil.copy2(A0 / "mapping.txt", OUT / "mapping.txt")

    for p in (A0 / "groundTruth").glob("case_*.txt"):
        q = OUT / "groundTruth" / p.name
        if not q.exists():
            shutil.copy2(p, q)

    for p in (A0 / "splits").glob("*.bundle"):
        q = OUT / "splits" / p.name
        if not q.exists():
            shutil.copy2(p, q)


def read_ids(bundle):
    ids = []
    with open(A0 / "splits" / bundle, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("case_") and s.endswith(".txt"):
                ids.append(int(s.replace("case_", "").replace(".txt", "")))
    return ids


def gt_len(vid):
    p = A0 / "groundTruth" / f"case_{vid}.txt"
    return sum(1 for _ in open(p, encoding="utf-8"))


def build_nfnet_x5(device):
    model = timm.create_model(
        "dm_nfnet_f0",
        pretrained=True,
        num_classes=0,
        in_chans=5,
    )
    model.eval().to(device)

    with torch.no_grad():
        test = torch.randn(2, 5, 224, 224).to(device)
        y = model(test)
        feat_dim = int(y.shape[1])

    print("NFNet output dim:", feat_dim)
    return model, feat_dim


def draw_rect_mask(mask, box):
    H, W = mask.shape
    x1, y1, x2, y2 = box.astype(int).tolist()
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 > x1 and y2 > y1:
        mask[y1:y2 + 1, x1:x2 + 1] = 1.0


def make_masks(frame, z, local_idx):
    H, W = frame.shape[:2]
    inst_mask = np.zeros((H, W), dtype=np.float32)
    eye_mask = np.zeros((H, W), dtype=np.float32)

    if z["eye_valid"][local_idx] > 0:
        draw_rect_mask(eye_mask, z["eye_xyxy"][local_idx])

    if "inst_xyxy_multi" in z and "inst_valid_multi" in z:
        boxes = z["inst_xyxy_multi"][local_idx]
        valids = z["inst_valid_multi"][local_idx]
        for k in range(boxes.shape[0]):
            if valids[k] > 0:
                draw_rect_mask(inst_mask, boxes[k])
    else:
        if z["inst_valid"][local_idx] > 0:
            draw_rect_mask(inst_mask, z["inst_xyxy"][local_idx])

    return inst_mask, eye_mask


def preprocess_x5_batch(frames, inst_masks, eye_masks):
    xs = []

    for frame, im, em in zip(frames, inst_masks, eye_masks):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        em = cv2.resize(em, (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        x = np.concatenate([rgb, im[..., None], em[..., None]], axis=2)
        xs.append(x)

    x = np.stack(xs).astype(np.float32)
    x = torch.from_numpy(x).permute(0, 3, 1, 2)

    # NFNet/timm pretrained uses RGB statistics; mask channels stay 0/1.
    mean = torch.tensor([0.485, 0.456, 0.406, 0.0, 0.0]).view(1, 5, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 1.0, 1.0]).view(1, 5, 1, 1)

    return (x - mean) / std


def extract_one(vid, model, feat_dim, device, batch_size=32, overwrite=False, save_vis=False, vis_stride=1500):
    out_path = OUT / "features" / f"case_{vid}.npy"
    meta_path = OUT / "meta" / f"case_{vid}_x5_nfnet.json"

    T = gt_len(vid)

    if out_path.exists() and not overwrite:
        arr = np.load(out_path, mmap_mode="r")
        if arr.shape == (feat_dim, T):
            print(f"[SKIP] case_{vid}: exists {arr.shape}")
            return

    yolo_path = YOLO / "npz" / f"case_{vid}.npz"
    if not yolo_path.exists():
        raise FileNotFoundError(yolo_path)

    z = np.load(yolo_path)
    frame_ids = z["frame_ids"]
    start = int(frame_ids[0])

    if len(frame_ids) != T:
        raise RuntimeError(f"case_{vid}: yolo T={len(frame_ids)} != gt T={T}")

    video_path = CAT / "videos" / f"case_{vid}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if save_vis:
        vis_dir = OUT / "meta" / "x5_vis" / f"case_{vid}"
        vis_dir.mkdir(parents=True, exist_ok=True)
    else:
        vis_dir = None

    feats = []
    frames, inst_masks, eye_masks = [], [], []
    idx = 0
    used = 0

    inst_area_sum = 0.0
    eye_area_sum = 0.0

    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx >= start and used < T:
                im, em = make_masks(frame, z, used)

                inst_area_sum += float(im.mean())
                eye_area_sum += float(em.mean())

                frames.append(frame)
                inst_masks.append(im)
                eye_masks.append(em)

                if save_vis and used % vis_stride == 0:
                    overlay = frame.copy()
                    overlay[im > 0] = (0, 0, 255)
                    overlay[em > 0] = (0, 255, 0)
                    vis = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
                    cv2.imwrite(str(vis_dir / f"frame_{idx:06d}.jpg"), vis)

                used += 1

                if len(frames) == batch_size:
                    x = preprocess_x5_batch(frames, inst_masks, eye_masks).to(device, non_blocking=True)
                    y = model(x).detach().float().cpu().numpy()
                    feats.append(y)
                    frames, inst_masks, eye_masks = [], [], []

            idx += 1
            if used >= T:
                break

        if frames:
            x = preprocess_x5_batch(frames, inst_masks, eye_masks).to(device, non_blocking=True)
            y = model(x).detach().float().cpu().numpy()
            feats.append(y)

    cap.release()

    if used != T:
        raise RuntimeError(f"case_{vid}: used={used}, expected T={T}")

    feat = np.concatenate(feats, axis=0).astype(np.float32).T

    if feat.shape != (feat_dim, T):
        raise RuntimeError(f"case_{vid}: feature shape {feat.shape}, expected ({feat_dim},{T})")

    np.save(out_path, feat)

    meta = {
        "video_id": int(vid),
        "T": int(T),
        "start_frame": int(start),
        "feature_shape": list(feat.shape),
        "feature_dim": int(feat_dim),
        "backbone": "dm_nfnet_f0",
        "input": "RGB + inst_mask_union_top2 + eye_mask",
        "inst_mask_area_mean": inst_area_sum / T,
        "eye_mask_area_mean": eye_area_sum / T,
        "yolo_path": str(yolo_path),
        "out_path": str(out_path),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] case_{vid}: feat={feat.shape}, "
        f"inst_area={meta['inst_mask_area_mean']:.4f}, "
        f"eye_area={meta['eye_mask_area_mean']:.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", type=int, default=None)
    ap.add_argument("--bundle", default="all.bundle")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save_vis", type=int, default=0)
    ap.add_argument("--vis_stride", type=int, default=1500)
    args = ap.parse_args()

    prepare_out()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model, feat_dim = build_nfnet_x5(device)

    vids = [args.video_id] if args.video_id is not None else read_ids(args.bundle)
    print("num videos:", len(vids))

    for i, vid in enumerate(vids, 1):
        print(f"\n===== [{i}/{len(vids)}] case_{vid} =====")
        extract_one(
            vid=vid,
            model=model,
            feat_dim=feat_dim,
            device=device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            save_vis=bool(args.save_vis),
            vis_stride=args.vis_stride,
        )


if __name__ == "__main__":
    main()

import os
# -*- coding: utf-8 -*-
import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))
CAT = Path(os.environ.get("CATARACT101_ROOT", "/path/to/cataract-101"))
BASE_DATA = ROOT / "data/cataract101_m10_a0_rgb2048"
MASKD = ROOT / "data/cataract101_m10_a2c_sam2_masks_full"
OLD_TRAIN_PY = ROOT / "m8/7_train_m8_igcoattn_v1_2_dualgate_framecls.py"


def log(x):
    print(x, flush=True)


def load_old_module():
    spec = importlib.util.spec_from_file_location("m8_v12", str(OLD_TRAIN_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_num_classes():
    p = BASE_DATA / "mapping.txt"
    ids = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].isdigit():
            ids.append(int(parts[0]))
        elif parts[-1].isdigit():
            ids.append(int(parts[-1]))
        else:
            ids.append(len(ids))
    return len(set(ids))


def prepare_out(out_data):
    out_data.mkdir(parents=True, exist_ok=True)
    (out_data / "features").mkdir(parents=True, exist_ok=True)

    for sub in ["groundTruth", "splits"]:
        src = BASE_DATA / sub
        dst = out_data / sub
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*"):
            q = dst / p.name
            if not q.exists():
                shutil.copy2(p, q)

    shutil.copy2(BASE_DATA / "mapping.txt", out_data / "mapping.txt")


def get_video_ids():
    vids = []
    for p in sorted((BASE_DATA / "groundTruth").glob("case_*.txt")):
        vids.append(int(p.stem.replace("case_", "")))
    return vids


def gt_len(video_id):
    p = BASE_DATA / "groundTruth" / f"case_{video_id}.txt"
    return sum(1 for _ in open(p, encoding="utf-8"))


def unpack_masks_batch(packed_batch, h=224, w=224):
    bits = np.unpackbits(packed_batch, axis=1, bitorder="little")[:, :h * w]
    return bits.reshape(-1, h, w).astype(np.float32)


def preprocess_x5(frames, inst_masks, eye_masks):
    rgbs = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        rgbs.append(rgb)

    rgb_np = np.stack(rgbs).astype(np.float32)
    inst_np = inst_masks[..., None].astype(np.float32)
    eye_np = eye_masks[..., None].astype(np.float32)

    x = np.concatenate([rgb_np, inst_np, eye_np], axis=3)
    x = torch.from_numpy(x).permute(0, 3, 1, 2)

    mean = torch.tensor([0.485, 0.456, 0.406, 0.0, 0.0]).view(1, 5, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 1.0, 1.0]).view(1, 5, 1, 1)

    return (x - mean) / std


def build_model(ckpt_path, device):
    mod = load_old_module()
    model_cls = getattr(mod, "M8IGCoAttnV1Asym")
    num_classes = get_num_classes()

    model = model_cls(
        backbone_name="dm_nfnet_f0",
        num_classes=num_classes,
        pretrained=True,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"[CKPT] loaded: {ckpt_path}")
    log(f"[CKPT] missing={len(missing)}, unexpected={len(unexpected)}")
    log(f"[CKPT] epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else None}")
    log(f"[CKPT] best_acc={ckpt.get('best_acc') if isinstance(ckpt, dict) else None}")

    model.eval().to(device)

    captured = {}

    def hook_fn(module, inp, out):
        captured["token"] = inp[0].detach()

    target = None
    for name, module in model.named_modules():
        if name.endswith("classifier"):
            target = (name, module)

    if target is None:
        raise RuntimeError("Cannot find classifier module for token hook.")

    name, module = target
    log(f"[HOOK] target: {name} -> {module.__class__.__name__}")
    handle = module.register_forward_hook(hook_fn)

    return model, captured, handle


def extract_one(video_id, model, captured, device, out_data, batch_size=32, overwrite=False, progress_every=3000):
    out_path = out_data / "features" / f"case_{video_id}.npy"
    T = gt_len(video_id)

    if out_path.exists() and not overwrite:
        arr = np.load(out_path, mmap_mode="r")
        if arr.shape == (384, T):
            log(f"[SKIP] case_{video_id}: exists {arr.shape}")
            return

    mask_path = MASKD / "npz" / f"case_{video_id}.npz"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    z = np.load(mask_path)
    frame_ids = z["frame_ids"]
    mask_h = int(z["mask_h"])
    mask_w = int(z["mask_w"])

    if len(frame_ids) != T:
        raise RuntimeError(f"case_{video_id}: mask T={len(frame_ids)} != gt T={T}")

    video_path = CAT / "videos" / f"case_{video_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    feats = []
    frames = []
    local_indices = []
    used = 0

    consecutive = bool(np.all(np.diff(frame_ids) == 1)) if len(frame_ids) > 1 else True
    if consecutive:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_ids[0]))

    def flush_batch():
        nonlocal frames, local_indices

        if not frames:
            return

        idx_arr = np.asarray(local_indices, dtype=np.int64)

        eye_masks = unpack_masks_batch(z["eye_mask_packed"][idx_arr], mask_h, mask_w)
        inst_masks = unpack_masks_batch(z["inst_union_mask_packed"][idx_arr], mask_h, mask_w)

        x = preprocess_x5(frames, inst_masks, eye_masks).to(device, non_blocking=True)

        captured.clear()
        with torch.inference_mode():
            _logits, _attn = model(x, return_attn=True)

        if "token" not in captured:
            raise RuntimeError("Failed to capture 384D token from classifier hook.")

        token = captured["token"].detach().float().cpu().numpy()

        if token.ndim != 2 or token.shape[1] != 384:
            raise RuntimeError(f"Unexpected token shape: {token.shape}")

        feats.append(token.astype(np.float32))

        frames = []
        local_indices = []

    for local_idx in range(T):
        gid = int(frame_ids[local_idx])

        if not consecutive:
            cap.set(cv2.CAP_PROP_POS_FRAMES, gid)

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"case_{video_id}: cannot read frame {gid}")

        frames.append(frame)
        local_indices.append(local_idx)
        used += 1

        if len(frames) >= batch_size:
            flush_batch()

        if used % progress_every == 0:
            log(f"[PROGRESS] case_{video_id}: {used}/{T}")

    flush_batch()
    cap.release()

    feat = np.concatenate(feats, axis=0).astype(np.float32).T

    if feat.shape != (384, T):
        raise RuntimeError(f"case_{video_id}: feat shape {feat.shape}, expected (384,{T})")

    np.save(out_path, feat)
    log(f"[OK] case_{video_id}: saved {feat.shape} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dataset", type=str, default="cataract101_m10_a3b_v12_dualgate_s1_best")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--video_id", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_data = ROOT / "data" / args.out_dataset
    prepare_out(out_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")
    log(f"out_dataset: {args.out_dataset}")
    log(f"out_data: {out_data}")

    model, captured, handle = build_model(args.ckpt, device)

    vids = [args.video_id] if args.video_id is not None else get_video_ids()
    log(f"num videos: {len(vids)}")

    for i, vid in enumerate(vids, 1):
        log(f"\n===== [{i}/{len(vids)}] case_{vid} =====")
        extract_one(
            video_id=vid,
            model=model,
            captured=captured,
            device=device,
            out_data=out_data,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )

    handle.remove()

    summary = {
        "out_dataset": args.out_dataset,
        "ckpt": args.ckpt,
        "feature_dim": 384,
        "num_videos": len(vids),
    }
    with open(out_data / "a3b_v12_feature_extract_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log("[DONE] feature extraction finished.")


if __name__ == "__main__":
    main()

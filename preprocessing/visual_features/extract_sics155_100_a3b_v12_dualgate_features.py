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

# SICS official 100-video dataset
DATA = ROOT / "data/sics155_100_geoig_r025"
MANIFEST = DATA / "frame_manifest"
MASKD = ROOT / "data/sics155_100_sam2_masks_full"

# Keep Cat101 base only for checkpoint-compatible classifier class/num_classes.
# Do NOT use SICS num_classes here, because the loaded A3B checkpoint is Cat101-trained.
CAT101_BASE = ROOT / "data/cataract101_m10_a0_rgb2048"

OLD_TRAIN_PY = ROOT / "m8/7_train_m8_igcoattn_v1_2_dualgate_framecls.py"

OUT_DATA = DATA
EXPECTED_DIM = 384


def log(x):
    print(x, flush=True)


def load_old_module():
    spec = importlib.util.spec_from_file_location("m8_v12", str(OLD_TRAIN_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_cat101_num_classes():
    p = CAT101_BASE / "mapping.txt"
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

    # Ensure standard MS-TCN2 files remain present
    for sub in ["groundTruth", "splits", "frame_manifest", "reports"]:
        src = DATA / sub
        dst = out_data / sub
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*"):
            q = dst / p.name
            if p.is_file() and not q.exists():
                shutil.copy2(p, q)

    if (DATA / "mapping.txt").exists():
        q = out_data / "mapping.txt"
        if not q.exists():
            shutil.copy2(DATA / "mapping.txt", q)


def read_bundle(bundle):
    if bundle in ["all", "all.bundle"]:
        return sorted([p.stem for p in (DATA / "groundTruth").glob("*.txt")])

    p = DATA / "splits" / bundle
    if not p.exists():
        raise FileNotFoundError(p)

    videos = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith(".txt"):
            s = s[:-4]
        videos.append(s)
    return videos


def gt_len(video):
    p = DATA / "groundTruth" / f"{video}.txt"
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def read_manifest(video):
    p = MANIFEST / f"{video}.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def get_packed_key(z, candidates):
    for k in candidates:
        if k in z:
            return k
    raise KeyError(f"Cannot find any key in {candidates}. Existing keys={list(z.keys())}")


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

    # Important: keep Cat101 class count so checkpoint classifier shape matches.
    num_classes = get_cat101_num_classes()

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
    log(f"[MODEL] checkpoint-compatible num_classes={num_classes}")

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


def extract_one(video, model, captured, device, out_data, batch_size=32, overwrite=False, progress_every=3000):
    out_path = out_data / "features" / f"{video}.npy"
    T = gt_len(video)
    frame_paths = read_manifest(video)

    if len(frame_paths) != T:
        raise RuntimeError(f"{video}: manifest T={len(frame_paths)} != gt T={T}")

    if out_path.exists() and not overwrite:
        arr = np.load(out_path, mmap_mode="r")
        if arr.shape == (EXPECTED_DIM, T):
            log(f"[SKIP] {video}: exists {arr.shape}")
            return {
                "video": video,
                "status": "skip",
                "shape": list(arr.shape),
                "path": str(out_path),
            }

    mask_path = MASKD / "npz" / f"{video}.npz"
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    z = np.load(mask_path, allow_pickle=True)
    frame_ids = z["frame_ids"]
    mask_h = int(z["mask_h"])
    mask_w = int(z["mask_w"])

    if len(frame_ids) != T:
        raise RuntimeError(f"{video}: mask T={len(frame_ids)} != gt T={T}")

    eye_key = get_packed_key(z, ["eye_mask_packed", "eye_masks_packed", "eye_packed"])
    inst_key = get_packed_key(z, ["inst_union_mask_packed", "inst_masks_packed", "inst_packed"])

    eye_packed = z[eye_key]
    inst_packed = z[inst_key]

    if eye_packed.shape[0] != T or inst_packed.shape[0] != T:
        raise RuntimeError(
            f"{video}: packed mask shape mismatch, "
            f"eye={eye_packed.shape}, inst={inst_packed.shape}, T={T}"
        )

    feats = []

    frames = []
    local_indices = []

    def flush():
        nonlocal frames, local_indices

        if not frames:
            return

        eye_masks = unpack_masks_batch(eye_packed[local_indices], h=mask_h, w=mask_w)
        inst_masks = unpack_masks_batch(inst_packed[local_indices], h=mask_h, w=mask_w)

        x = preprocess_x5(frames, inst_masks, eye_masks).to(device, non_blocking=True)

        captured.clear()
        with torch.inference_mode():
            _logits, _attn = model(x, return_attn=True)

        if "token" not in captured:
            raise RuntimeError(f"{video}: Failed to capture 384D token from classifier hook.")

        token = captured["token"].detach().float().cpu().numpy()

        if token.ndim != 2 or token.shape[1] != EXPECTED_DIM:
            raise RuntimeError(f"{video}: Unexpected token shape: {token.shape}")

        token_np = token.astype(np.float32)

        if token_np.shape[0] != len(frames):
            raise RuntimeError(f"{video}: token batch mismatch {token_np.shape}, batch={len(frames)}")

        feats.append(token_np)

        frames = []
        local_indices = []

    for t, fp in enumerate(frame_paths):
        frame = cv2.imread(fp)
        if frame is None:
            raise RuntimeError(f"{video}: cannot read t={t}, path={fp}")

        frames.append(frame)
        local_indices.append(t)

        if len(frames) >= batch_size:
            flush()

        if progress_every > 0 and (t + 1) % progress_every == 0:
            log(f"[PROGRESS] {video}: {t+1}/{T}")

    flush()

    feat_tdim = np.concatenate(feats, axis=0).astype(np.float32)

    if feat_tdim.shape[0] != T:
        raise RuntimeError(f"{video}: feat T={feat_tdim.shape[0]} != {T}")

    if feat_tdim.shape[1] != EXPECTED_DIM:
        raise RuntimeError(f"{video}: feature dim={feat_tdim.shape[1]}, expected {EXPECTED_DIM}")

    feat = feat_tdim.T.copy()

    if feat.shape != (EXPECTED_DIM, T):
        raise RuntimeError(f"{video}: feat shape {feat.shape}, expected ({EXPECTED_DIM},{T})")

    np.save(out_path, feat)

    log(f"[OK] {video}: saved {feat.shape} -> {out_path}")

    return {
        "video": video,
        "status": "ok",
        "shape": list(feat.shape),
        "path": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dataset", type=str, default="sics155_100_geoig_r025")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--bundle", type=str, default="all")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--progress_every", type=int, default=3000)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_data = ROOT / "data" / args.out_dataset
    prepare_out(out_data)

    log("=" * 80)
    log("[SICS155_100 A3B v12 DUALGATE FEATURE EXTRACTION]")
    log(f"out_dataset: {args.out_dataset}")
    log(f"out_data: {out_data}")
    log(f"mask_dir: {MASKD}")
    log(f"ckpt: {args.ckpt}")
    log(f"device: {device}")
    log("=" * 80)

    model, captured, handle = build_model(args.ckpt, device)

    if args.video:
        videos = [args.video]
    else:
        videos = read_bundle(args.bundle)

    if args.limit and args.limit > 0:
        videos = videos[:args.limit]

    log(f"num videos: {len(videos)}")
    log(f"videos: {videos[:10]} ...")

    summary = []

    for i, video in enumerate(videos, start=1):
        log(f"\n===== [{i}/{len(videos)}] {video} =====")
        try:
            res = extract_one(
                video=video,
                model=model,
                captured=captured,
                device=device,
                out_data=out_data,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                progress_every=args.progress_every,
            )
        except Exception as e:
            res = {
                "video": video,
                "status": "error",
                "error": repr(e),
            }
            log("[ERROR] " + json.dumps(res, ensure_ascii=False))

        summary.append(res)

        with open(out_data / "a3b_v12_feature_extract_summary.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    ok = sum(1 for x in summary if x.get("status") in ["ok", "skip"])
    err = sum(1 for x in summary if x.get("status") == "error")

    final = {
        "out_dataset": args.out_dataset,
        "ckpt": args.ckpt,
        "num_videos": len(videos),
        "ok_or_skip": ok,
        "error": err,
    }

    with open(out_data / "a3b_v12_feature_extract_summary.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    handle.remove()

    log("[DONE]")
    log(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

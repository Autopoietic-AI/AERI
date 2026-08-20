import os
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))

DATA = ROOT / "data/sics155_100_geoig_r025"
SAM2 = ROOT / "data/sics155_100_sam2_masks_full/npz"
YOLO = ROOT / "data/sics155_100_yolo_b2_multiinst_assets/npz"
A4 = ROOT / "data/sics155_100_a4_geo_assets_sam2_yolo_tiptrack_cons_compat/npz"

OUT_ROOT = ROOT / "data/sics155_100_a5_maskshape_assets_v1_safe"
OUT_NPZ = OUT_ROOT / "npz"
OUT_META = OUT_ROOT / "meta"

MASK_SHAPE_NAMES = [
    "inst_area_ratio",          # 0
    "inst_area_delta",          # 1
    "union_area_ratio",         # 2
    "union_area_delta",         # 3
    "inst_score",               # 4
    "eye_score",                # 5
    "inst_valid",               # 6
    "eye_valid",                # 7
    "center_dx_norm",           # 8
    "center_dy_norm",           # 9
    "center_dist_norm",         # 10
    "center_dist_delta",        # 11
    "union_dx_norm",            # 12
    "union_dy_norm",            # 13
    "union_center_dist_norm",   # 14, used by A5 q_idx
    "union_bbox_eye_ratio",     # 15
]

CLIP_RANGES = {
    "inst_area_ratio": (0.0, 0.12),
    "inst_area_delta": (-0.05, 0.05),
    "union_area_ratio": (0.0, 0.20),
    "union_area_delta": (-0.08, 0.08),
    "inst_score": (0.0, 1.0),
    "eye_score": (0.0, 1.0),
    "inst_valid": (0.0, 1.0),
    "eye_valid": (0.0, 1.0),
    "center_dx_norm": (-6.0, 6.0),
    "center_dy_norm": (-6.0, 6.0),
    "center_dist_norm": (0.0, 8.0),
    "center_dist_delta": (-4.0, 4.0),
    "union_dx_norm": (-6.0, 6.0),
    "union_dy_norm": (-6.0, 6.0),
    "union_center_dist_norm": (0.0, 8.0),
    "union_bbox_eye_ratio": (0.0, 3.0),
}

EPS = 1e-6
MAX_INST = 2


def prepare_out():
    for sub in ["npz", "meta", "groundTruth", "splits"]:
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    shutil.copy2(DATA / "mapping.txt", OUT_ROOT / "mapping.txt")

    for p in (DATA / "groundTruth").glob("*.txt"):
        q = OUT_ROOT / "groundTruth" / p.name
        if not q.exists():
            shutil.copy2(p, q)

    for p in (DATA / "splits").glob("*.bundle"):
        q = OUT_ROOT / "splits" / p.name
        if not q.exists():
            shutil.copy2(p, q)


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
    return sum(1 for x in open(DATA / "groundTruth" / f"{video}.txt", encoding="utf-8") if x.strip())


def get_key(z, candidates):
    for k in candidates:
        if k in z:
            return k
    raise KeyError(f"Cannot find any key in {candidates}; existing={list(z.keys())}")


def safe_delta(x):
    y = np.zeros_like(x, dtype=np.float32)
    if len(x) > 1:
        y[1:] = x[1:] - x[:-1]
    return y


def unpack_stats(packed, h, w, chunk=512, bitorder="little"):
    """
    Compute area, center, bbox area and compactness from packed binary masks.
    packed: (T, pack_len)
    returns:
      area_px, cx, cy, bbox_area_px, compactness
    """
    T = packed.shape[0]
    area = np.zeros(T, dtype=np.float32)
    cx = np.zeros(T, dtype=np.float32)
    cy = np.zeros(T, dtype=np.float32)
    bbox_area = np.zeros(T, dtype=np.float32)
    compactness = np.zeros(T, dtype=np.float32)

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)

    hw = h * w

    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        bits = np.unpackbits(packed[s:e], axis=1, bitorder=bitorder)[:, :hw]
        m = bits.reshape(e - s, h, w).astype(bool)

        a = m.sum(axis=(1, 2)).astype(np.float32)
        area[s:e] = a

        nonempty = a > 0

        row_sum = m.sum(axis=2).astype(np.float32)  # (N,H)
        col_sum = m.sum(axis=1).astype(np.float32)  # (N,W)

        cx_chunk = (col_sum * xs[None, :]).sum(axis=1) / np.maximum(a, 1.0)
        cy_chunk = (row_sum * ys[None, :]).sum(axis=1) / np.maximum(a, 1.0)

        cx[s:e] = np.where(nonempty, cx_chunk, 0.0)
        cy[s:e] = np.where(nonempty, cy_chunk, 0.0)

        row_any = m.any(axis=2)
        col_any = m.any(axis=1)

        first_y = np.argmax(row_any, axis=1)
        last_y = h - 1 - np.argmax(row_any[:, ::-1], axis=1)
        first_x = np.argmax(col_any, axis=1)
        last_x = w - 1 - np.argmax(col_any[:, ::-1], axis=1)

        ba = ((last_y - first_y + 1) * (last_x - first_x + 1)).astype(np.float32)
        ba = np.where(nonempty, ba, 0.0)

        bbox_area[s:e] = ba
        compactness[s:e] = np.where(nonempty, a / np.maximum(ba, 1.0), 0.0)

    return area, cx, cy, bbox_area, compactness


def build_one(video, overwrite=False, chunk=512, bitorder="little"):
    out_path = OUT_NPZ / f"{video}.npz"
    meta_path = OUT_META / f"{video}_maskshape_summary.json"

    T_gt = gt_len(video)

    if out_path.exists() and meta_path.exists() and not overwrite:
        z0 = np.load(out_path)
        if "mask_shape" in z0 and z0["mask_shape"].shape[-1] == T_gt:
            print(f"[SKIP] {video}: exists", flush=True)
            return {"video": video, "status": "skip", "T": int(T_gt), "npz": str(out_path)}

    sam = np.load(SAM2 / f"{video}.npz", allow_pickle=True)
    a4 = np.load(A4 / f"{video}.npz", allow_pickle=True)
    yolo = np.load(YOLO / f"{video}.npz", allow_pickle=True)

    frame_ids = sam["frame_ids"].astype(np.int32)
    T = len(frame_ids)

    if T != T_gt:
        raise RuntimeError(f"{video}: SAM2 T={T}, GT T={T_gt}")

    mask_h = int(sam["mask_h"])
    mask_w = int(sam["mask_w"])
    hw = float(mask_h * mask_w)

    eye_key = get_key(sam, ["eye_mask_packed", "eye_masks_packed", "eye_packed"])
    inst_key = get_key(sam, ["inst_union_mask_packed", "inst_masks_packed", "inst_packed"])

    eye_packed = sam[eye_key]
    inst_packed = sam[inst_key]

    eye_area_px, eye_cx, eye_cy, eye_bbox_area_px, eye_compactness = unpack_stats(
        eye_packed, mask_h, mask_w, chunk=chunk, bitorder=bitorder
    )

    union_area_px, union_cx, union_cy, union_bbox_area_px, union_compactness = unpack_stats(
        inst_packed, mask_h, mask_w, chunk=chunk, bitorder=bitorder
    )

    eye_area_ratio = eye_area_px / hw
    union_area_ratio = union_area_px / hw
    union_area_delta = safe_delta(union_area_ratio)

    eye_radius_px = np.sqrt(np.maximum(eye_area_px, 1.0) / np.pi).astype(np.float32)
    eye_radius_safe = np.maximum(eye_radius_px, 1.0)

    # A4 has inst_centers: (2,2,T), normalized by image W/H.
    inst_centers = a4["inst_centers"].astype(np.float32)  # (2,2,T)
    inst_centers_t = inst_centers.transpose(2, 0, 1)      # (T,2,2)

    inst_center_px = np.zeros_like(inst_centers_t, dtype=np.float32)
    inst_center_px[:, :, 0] = inst_centers_t[:, :, 0] * max(mask_w - 1, 1)
    inst_center_px[:, :, 1] = inst_centers_t[:, :, 1] * max(mask_h - 1, 1)

    inst_valid_multi = a4["inst_valid_multi"].astype(np.float32)[:T, :MAX_INST]
    if inst_valid_multi.shape[1] < MAX_INST:
        pad = np.zeros((T, MAX_INST - inst_valid_multi.shape[1]), dtype=np.float32)
        inst_valid_multi = np.concatenate([inst_valid_multi, pad], axis=1)

    eye_valid = a4["eye_valid"].astype(np.float32)[:T]

    # A4 inst_area_multi is ratio-like. If abnormal, convert defensively.
    inst_area_multi = a4["inst_area_multi"].astype(np.float32)[:T, :MAX_INST]
    if inst_area_multi.shape[1] < MAX_INST:
        pad = np.zeros((T, MAX_INST - inst_area_multi.shape[1]), dtype=np.float32)
        inst_area_multi = np.concatenate([inst_area_multi, pad], axis=1)
    if np.nanpercentile(inst_area_multi, 95) > 10:
        inst_area_multi = inst_area_multi / hw

    if "inst_conf_multi" in yolo:
        inst_score_multi = yolo["inst_conf_multi"].astype(np.float32)[:T, :MAX_INST]
    else:
        inst_score_multi = inst_valid_multi.copy()

    if inst_score_multi.shape[1] < MAX_INST:
        pad = np.zeros((T, MAX_INST - inst_score_multi.shape[1]), dtype=np.float32)
        inst_score_multi = np.concatenate([inst_score_multi, pad], axis=1)

    eye_score = yolo["eye_conf"].astype(np.float32)[:T] if "eye_conf" in yolo else eye_valid

    union_dx = (union_cx - eye_cx) / eye_radius_safe
    union_dy = (union_cy - eye_cy) / eye_radius_safe
    union_center_dist = np.sqrt(union_dx * union_dx + union_dy * union_dy)
    union_bbox_eye_ratio = union_bbox_area_px / np.maximum(eye_area_px, 1.0)

    D = len(MASK_SHAPE_NAMES)
    mask_shape = np.zeros((MAX_INST, D, T), dtype=np.float32)
    mask_shape_valid = np.zeros((MAX_INST, T), dtype=np.float32)

    for k in range(MAX_INST):
        inst_area_ratio = inst_area_multi[:, k]
        inst_area_delta = safe_delta(inst_area_ratio)

        cx = inst_center_px[:, k, 0]
        cy = inst_center_px[:, k, 1]

        dx = (cx - eye_cx) / eye_radius_safe
        dy = (cy - eye_cy) / eye_radius_safe
        dist = np.sqrt(dx * dx + dy * dy)
        dist_delta = safe_delta(dist)

        vals = [
            inst_area_ratio,
            inst_area_delta,
            union_area_ratio,
            union_area_delta,
            inst_score_multi[:, k],
            eye_score,
            inst_valid_multi[:, k],
            eye_valid,
            dx,
            dy,
            dist,
            dist_delta,
            union_dx,
            union_dy,
            union_center_dist,
            union_bbox_eye_ratio,
        ]

        for i, v in enumerate(vals):
            x = np.nan_to_num(v.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            lo, hi = CLIP_RANGES[MASK_SHAPE_NAMES[i]]
            x = np.clip(x, lo, hi)
            mask_shape[k, i, :] = x

        mask_shape_valid[k] = ((inst_valid_multi[:, k] > 0.5) & (eye_valid > 0.5)).astype(np.float32)

    eye_area_saved = a4["eye_area"].astype(np.float32)[:T] if "eye_area" in a4 else sam["eye_area"].astype(np.float32)[:T]
    union_area_saved = a4["inst_area_union"].astype(np.float32)[:T] if "inst_area_union" in a4 else sam["inst_area"].astype(np.float32)[:T]

    eye_area_abs_err = np.abs(eye_area_ratio - eye_area_saved)
    union_area_abs_err = np.abs(union_area_ratio - union_area_saved)

    np.savez_compressed(
        out_path,
        frame_ids=frame_ids.astype(np.int32),
        mask_h=np.array(mask_h, dtype=np.int32),
        mask_w=np.array(mask_w, dtype=np.int32),
        mask_shape=mask_shape.astype(np.float32),
        mask_shape_valid=mask_shape_valid.astype(np.float32),
        mask_shape_names=np.asarray(MASK_SHAPE_NAMES),

        eye_area_px=eye_area_px.astype(np.float32),
        eye_area_ratio=eye_area_ratio.astype(np.float32),
        eye_cx=eye_cx.astype(np.float32),
        eye_cy=eye_cy.astype(np.float32),

        union_area_px=union_area_px.astype(np.float32),
        union_area_ratio=union_area_ratio.astype(np.float32),
        union_cx=union_cx.astype(np.float32),
        union_cy=union_cy.astype(np.float32),
        union_compactness=union_compactness.astype(np.float32),
        union_bbox_area_px=union_bbox_area_px.astype(np.float32),
        union_bbox_eye_ratio=union_bbox_eye_ratio.astype(np.float32),

        eye_area_abs_err=eye_area_abs_err.astype(np.float32),
        union_area_abs_err=union_area_abs_err.astype(np.float32),
    )

    meta = {
        "video": video,
        "status": "ok",
        "T": int(T),
        "mask_shape": list(mask_shape.shape),
        "mask_shape_valid": list(mask_shape_valid.shape),
        "valid_mean": float(mask_shape_valid.mean()),
        "mask_shape_mean": float(mask_shape.mean()),
        "mask_shape_std": float(mask_shape.std()),
        "eye_area_abs_err_mean": float(eye_area_abs_err.mean()),
        "eye_area_abs_err_p95": float(np.percentile(eye_area_abs_err, 95)),
        "union_area_abs_err_mean": float(union_area_abs_err.mean()),
        "union_area_abs_err_p95": float(np.percentile(union_area_abs_err, 95)),
        "npz": str(out_path),
    }

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[OK] {video}: T={T}, mask_shape={mask_shape.shape}, "
        f"valid={meta['valid_mean']:.4f}, union_err={meta['union_area_abs_err_mean']:.6f}",
        flush=True,
    )

    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="all")
    ap.add_argument("--video", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--bitorder", default="little", choices=["little", "big"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prepare_out()

    if args.video:
        videos = [args.video]
    else:
        videos = read_bundle(args.bundle)

    if args.limit and args.limit > 0:
        videos = videos[:args.limit]

    print("=" * 80, flush=True)
    print("[SICS155_100 A5 maskshape assets v1_safe]", flush=True)
    print("DATA:", DATA, flush=True)
    print("SAM2:", SAM2, flush=True)
    print("A4:", A4, flush=True)
    print("YOLO:", YOLO, flush=True)
    print("OUT:", OUT_NPZ, flush=True)
    print("videos:", len(videos), videos[:10], "...", flush=True)
    print("bitorder:", args.bitorder, "chunk:", args.chunk, flush=True)
    print("=" * 80, flush=True)

    summary_path = OUT_META / "summary_all.jsonl"
    if args.overwrite and summary_path.exists():
        summary_path.unlink()

    rows = []

    for i, video in enumerate(videos, start=1):
        print(f"\n===== [{i}/{len(videos)}] {video} =====", flush=True)
        try:
            row = build_one(video, overwrite=args.overwrite, chunk=args.chunk, bitorder=args.bitorder)
        except Exception as e:
            row = {"video": video, "status": "error", "error": repr(e)}
            print("[ERROR]", json.dumps(row, ensure_ascii=False), flush=True)

        rows.append(row)
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = OUT_ROOT / "build_maskshape_summary.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print("[DONE]", flush=True)
    print("Output:", OUT_NPZ, flush=True)
    print("Summary:", summary_path, flush=True)
    print("CSV:", csv_path, flush=True)


if __name__ == "__main__":
    main()

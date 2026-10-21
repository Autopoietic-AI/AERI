# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import csv


MASK_SHAPE_NAMES = [
    "inst_area_ratio",
    "inst_area_delta",
    "union_area_ratio",
    "union_area_delta",
    "inst_score",
    "eye_score",
    "inst_valid",
    "eye_valid",
    "center_dx_norm",
    "center_dy_norm",
    "center_dist_norm",
    "union_dx_norm",
    "union_dy_norm",
    "union_dist_norm",
    "union_compactness",
    "union_bbox_eye_ratio",
]


def safe_delta(x):
    x = np.asarray(x, dtype=np.float32)
    d = np.zeros_like(x, dtype=np.float32)
    d[1:] = x[1:] - x[:-1]
    return d


def unpack_stats(packed, h, w, chunk=256, bitorder="little"):
    """
    从 packed binary mask 中计算:
    area_pixel, center_x, center_y, bbox_area_pixel, compactness
    """
    T = packed.shape[0]
    hw = int(h) * int(w)

    area = np.zeros(T, dtype=np.float32)
    cx = np.zeros(T, dtype=np.float32)
    cy = np.zeros(T, dtype=np.float32)
    bbox_area = np.zeros(T, dtype=np.float32)
    compactness = np.zeros(T, dtype=np.float32)

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)

    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        bits = np.unpackbits(packed[s:e], axis=1, bitorder=bitorder)
        bits = bits[:, :hw].reshape(e - s, h, w).astype(bool)

        a = bits.sum(axis=(1, 2)).astype(np.float32)
        area[s:e] = a

        nonempty = a > 0

        col_sum = bits.sum(axis=1).astype(np.float32)  # N,W
        row_sum = bits.sum(axis=2).astype(np.float32)  # N,H

        cx_i = (col_sum * xs[None, :]).sum(axis=1) / np.maximum(a, 1.0)
        cy_i = (row_sum * ys[None, :]).sum(axis=1) / np.maximum(a, 1.0)

        cx[s:e] = np.where(nonempty, cx_i, 0.0)
        cy[s:e] = np.where(nonempty, cy_i, 0.0)

        row_any = bits.any(axis=2)
        col_any = bits.any(axis=1)

        ymin = row_any.argmax(axis=1)
        ymax = h - 1 - row_any[:, ::-1].argmax(axis=1)
        xmin = col_any.argmax(axis=1)
        xmax = w - 1 - col_any[:, ::-1].argmax(axis=1)

        bw = np.maximum(0, xmax - xmin + 1).astype(np.float32)
        bh = np.maximum(0, ymax - ymin + 1).astype(np.float32)
        ba = bw * bh
        ba = np.where(nonempty, ba, 0.0)

        bbox_area[s:e] = ba
        compactness[s:e] = np.where(nonempty, a / np.maximum(ba, 1.0), 0.0)

    return area, cx, cy, bbox_area, compactness


def convert_center_to_pixel(center, h, w):
    """
    center: (..., 2)
    如果中心坐标大多在 0~1 之间，认为是归一化坐标；
    否则认为已经是像素坐标。
    """
    c = center.astype(np.float32).copy()
    finite = np.isfinite(c)

    if finite.any():
        p95 = np.percentile(np.abs(c[finite]), 95)
    else:
        p95 = 0.0

    if p95 <= 2.0:
        c[..., 0] = c[..., 0] * float(w)
        c[..., 1] = c[..., 1] * float(h)

    return c


def build_one(npz_path, out_path, chunk=256, bitorder="little"):
    z = np.load(npz_path, allow_pickle=True)

    frame_ids = z["frame_ids"].astype(np.int32)
    T = len(frame_ids)

    h = int(z["mask_h"])
    w = int(z["mask_w"])
    hw = float(h * w)

    # packed mask 重新计算，保证尺度一致
    eye_area_px, eye_cx, eye_cy, eye_bbox_area_px, eye_compactness = unpack_stats(
        z["eye_mask_packed"][:T], h, w, chunk=chunk, bitorder=bitorder
    )

    union_area_px, union_cx, union_cy, union_bbox_area_px, union_compactness = unpack_stats(
        z["inst_union_mask_packed"][:T], h, w, chunk=chunk, bitorder=bitorder
    )

    # 归一化面积比例
    eye_area_ratio = eye_area_px / hw
    union_area_ratio = union_area_px / hw

    # eye radius 使用像素面积计算
    eye_radius_px = np.sqrt(np.maximum(eye_area_px, 1.0) / np.pi).astype(np.float32)
    eye_radius_px = np.maximum(eye_radius_px, 1.0)

    # track-level 信息：来自 SAM2 资产中已有字段
    inst_area_multi_saved = z["inst_area_multi"].astype(np.float32)[:T]      # 通常是 ratio, (T,2)
    inst_center_multi = z["inst_center_multi"].astype(np.float32)[:T]        # (T,2,2)
    inst_center_px = convert_center_to_pixel(inst_center_multi, h, w)

    eye_valid = z["eye_valid"].astype(np.float32)[:T]
    inst_valid_multi = z["inst_valid_multi"].astype(np.float32)[:T]
    eye_score = z["eye_score"].astype(np.float32)[:T]
    inst_score_multi = z["inst_score_multi"].astype(np.float32)[:T]

    # 如果 inst_area_multi 已经是 0~1 ratio，直接使用；
    # 如果异常大，则当成 pixel area 转 ratio。
    inst_area_multi = inst_area_multi_saved.copy()
    if np.nanpercentile(inst_area_multi, 95) > 10:
        inst_area_ratio_multi = inst_area_multi / hw
    else:
        inst_area_ratio_multi = inst_area_multi

    # union 相对 eye 的位置
    union_dx = (union_cx - eye_cx) / eye_radius_px
    union_dy = (union_cy - eye_cy) / eye_radius_px
    union_dist = np.sqrt(union_dx ** 2 + union_dy ** 2)

    union_area_delta = safe_delta(union_area_ratio)

    # bbox area / eye area
    union_bbox_eye_ratio = union_bbox_area_px / np.maximum(eye_area_px, 1.0)

    D = len(MASK_SHAPE_NAMES)
    mask_shape = np.zeros((2, D, T), dtype=np.float32)
    mask_shape_valid = np.zeros((2, T), dtype=np.float32)

    for k in range(2):
        inst_area_ratio = inst_area_ratio_multi[:, k]
        inst_area_delta = safe_delta(inst_area_ratio)

        cx = inst_center_px[:, k, 0]
        cy = inst_center_px[:, k, 1]

        dx = (cx - eye_cx) / eye_radius_px
        dy = (cy - eye_cy) / eye_radius_px
        dist = np.sqrt(dx ** 2 + dy ** 2)

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
            union_dx,
            union_dy,
            union_dist,
            union_compactness,
            union_bbox_eye_ratio,
        ]

        for i, v in enumerate(vals):
            mask_shape[k, i] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        mask_shape_valid[k] = ((inst_valid_multi[:, k] > 0.5) & (eye_valid > 0.5)).astype(np.float32)

    # 温和裁剪，避免极少数 mask 异常污染模型
    for i, name in enumerate(MASK_SHAPE_NAMES):
        x = mask_shape[:, i, :]
        finite = x[np.isfinite(x)]
        if finite.size > 0:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            mask_shape[:, i, :] = np.clip(x, lo, hi)

    # 面积一致性：saved area 应该约等于 packed_area / hw
    saved_eye_area = z["eye_area"].astype(np.float32)[:T]
    saved_union_area = z["inst_area_union"].astype(np.float32)[:T]

    eye_area_abs_err = np.abs(eye_area_ratio - saved_eye_area)
    union_area_abs_err = np.abs(union_area_ratio - saved_union_area)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        frame_ids=frame_ids,
        mask_h=np.int32(h),
        mask_w=np.int32(w),
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

    return {
        "case": npz_path.stem,
        "T": T,
        "valid_mean": float(mask_shape_valid.mean()),
        "mask_shape_mean": float(mask_shape.mean()),
        "mask_shape_std": float(mask_shape.std()),
        "eye_area_abs_err_mean": float(eye_area_abs_err.mean()),
        "eye_area_abs_err_p95": float(np.percentile(eye_area_abs_err, 95)),
        "union_area_abs_err_mean": float(union_area_abs_err.mean()),
        "union_area_abs_err_p95": float(np.percentile(union_area_abs_err, 95)),
        "union_compactness_mean": float(union_compactness.mean()),
        "union_bbox_eye_ratio_mean": float(union_bbox_eye_ratio.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam2_dir", default="data/cataract101_m10_a2c_sam2_masks_full/npz")
    ap.add_argument("--out_dir", default="data/cataract101_m10_a5_maskshape_assets_v1/npz")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--bitorder", type=str, default="little", choices=["big", "little"])
    args = ap.parse_args()

    sam2_dir = Path(args.sam2_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(sam2_dir.glob("case_*.npz"), key=lambda p: int(p.stem.replace("case_", "")))
    print("[INFO] input files:", len(files))
    print("[INFO] out_dir:", out_dir)
    print("[INFO] bitorder:", args.bitorder)

    rows = []
    for i, p in enumerate(files, 1):
        row = build_one(p, out_dir / p.name, chunk=args.chunk, bitorder=args.bitorder)
        rows.append(row)

        if i <= 5 or i % 10 == 0:
            print(
                f"[{i:03d}/{len(files)}] {p.stem} "
                f"valid={row['valid_mean']:.3f} "
                f"shape_mean={row['mask_shape_mean']:.3f} "
                f"shape_std={row['mask_shape_std']:.3f} "
                f"union_err={row['union_area_abs_err_mean']:.6f}"
            )

    summary_csv = out_dir.parent / "build_maskshape_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("[SAVE]", summary_csv)
    print("[DONE] build mask-shape v1 assets finished.")


if __name__ == "__main__":
    main()

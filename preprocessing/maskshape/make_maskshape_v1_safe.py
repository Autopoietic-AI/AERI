# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


SAFE_RULES = {
    "inst_area_ratio":        (0.0, 0.12),
    "inst_area_delta":        (-0.05, 0.05),
    "union_area_ratio":       (0.0, 0.20),
    "union_area_delta":       (-0.08, 0.08),
    "inst_score":             (0.0, 1.0),
    "eye_score":              (0.0, 1.0),
    "inst_valid":             (0.0, 1.0),
    "eye_valid":              (0.0, 1.0),
    "center_dx_norm":         (-6.0, 6.0),
    "center_dy_norm":         (-6.0, 6.0),
    "center_dist_norm":       (0.0, 8.0),
    "union_dx_norm":          (-6.0, 6.0),
    "union_dy_norm":          (-6.0, 6.0),
    "union_dist_norm":        (0.0, 8.0),
    "union_compactness":      (0.0, 1.0),
}


def safe_one(src_path, dst_path):
    z = np.load(src_path, allow_pickle=True)

    x = z["mask_shape"].astype(np.float32).copy()       # (2,15,T)
    v = z["mask_shape_valid"].astype(np.float32).copy() # (2,T)
    names = [str(n) for n in z["mask_shape_names"]]

    before_min = float(x.min())
    before_max = float(x.max())
    before_std = float(x.std())

    clip_stats = {}

    for i, name in enumerate(names):
        lo, hi = SAFE_RULES.get(name, (-10.0, 10.0))
        xi = x[:, i, :]
        n_clip = int(((xi < lo) | (xi > hi)).sum())
        clip_stats[f"clip_{name}"] = n_clip
        x[:, i, :] = np.clip(xi, lo, hi)

    after_min = float(x.min())
    after_max = float(x.max())
    after_std = float(x.std())

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "frame_ids": z["frame_ids"],
        "mask_h": z["mask_h"],
        "mask_w": z["mask_w"],
        "mask_shape": x.astype(np.float32),
        "mask_shape_valid": v.astype(np.float32),
        "mask_shape_names": np.asarray(names),
    }

    for k in z.files:
        if k not in save_dict and k not in ["mask_shape", "mask_shape_valid", "mask_shape_names"]:
            save_dict[k] = z[k]

    np.savez_compressed(dst_path, **save_dict)

    row = {
        "case": src_path.stem,
        "T": x.shape[-1],
        "D": x.shape[1],
        "valid_mean": float(v.mean()),
        "mean": float(x.mean()),
        "std": after_std,
        "min": after_min,
        "max": after_max,
        "before_std": before_std,
        "before_min": before_min,
        "before_max": before_max,
        "total_clipped": int(sum(clip_stats.values())),
    }
    row.update(clip_stats)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/cataract101_m10_a5_maskshape_assets_v1_clean/npz")
    ap.add_argument("--out_dir", default="data/cataract101_m10_a5_maskshape_assets_v1_safe/npz")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("case_*.npz"), key=lambda p: int(p.stem.replace("case_", "")))
    print("[INFO] input files:", len(files))
    print("[INFO] out_dir:", out_dir)

    rows = []
    for i, p in enumerate(files, 1):
        row = safe_one(p, out_dir / p.name)
        rows.append(row)

        if i <= 5 or i % 10 == 0:
            print(
                f"[{i:03d}/{len(files)}] {p.stem} "
                f"valid={row['valid_mean']:.3f} "
                f"std {row['before_std']:.3f}->{row['std']:.3f} "
                f"range [{row['before_min']:.2f},{row['before_max']:.2f}]"
                f" -> [{row['min']:.2f},{row['max']:.2f}] "
                f"clipped={row['total_clipped']}"
            )

    summary = pd.DataFrame(rows)
    out_csv = out_dir.parent / "build_maskshape_v1_safe_summary.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("[SAVE]", out_csv)
    print("[DONE] maskshape v1 safe finished.")


if __name__ == "__main__":
    main()

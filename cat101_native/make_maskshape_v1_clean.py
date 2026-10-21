# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def clean_one(src_path, dst_path, drop_names=("union_bbox_eye_ratio",)):
    z = np.load(src_path, allow_pickle=True)

    x = z["mask_shape"].astype(np.float32)          # (2, D, T)
    v = z["mask_shape_valid"].astype(np.float32)    # (2, T)
    names = [str(n) for n in z["mask_shape_names"]]

    keep_idx = [i for i, n in enumerate(names) if n not in drop_names]
    keep_names = [names[i] for i in keep_idx]

    x_clean = x[:, keep_idx, :].copy()

    # 二次保险：对每个维度做温和 clip，避免极少数异常值污染模型
    for i, name in enumerate(keep_names):
        xi = x_clean[:, i, :]
        finite = xi[np.isfinite(xi)]
        if finite.size > 0:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            x_clean[:, i, :] = np.clip(xi, lo, hi)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # 保留原始辅助字段，方便后面诊断
    save_dict = {
        "frame_ids": z["frame_ids"],
        "mask_h": z["mask_h"],
        "mask_w": z["mask_w"],
        "mask_shape": x_clean.astype(np.float32),
        "mask_shape_valid": v.astype(np.float32),
        "mask_shape_names": np.asarray(keep_names),
    }

    for k in z.files:
        if k not in save_dict and k not in ["mask_shape", "mask_shape_valid", "mask_shape_names"]:
            save_dict[k] = z[k]

    np.savez_compressed(dst_path, **save_dict)

    return {
        "case": src_path.stem,
        "T": x.shape[-1],
        "D_old": x.shape[1],
        "D_new": x_clean.shape[1],
        "valid_mean": float(v.mean()),
        "mean": float(x_clean.mean()),
        "std": float(x_clean.std()),
        "min": float(x_clean.min()),
        "max": float(x_clean.max()),
        "names": ",".join(keep_names),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/cataract101_m10_a5_maskshape_assets_v1/npz")
    ap.add_argument("--out_dir", default="data/cataract101_m10_a5_maskshape_assets_v1_clean/npz")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("case_*.npz"), key=lambda p: int(p.stem.replace("case_", "")))
    print("[INFO] input files:", len(files))
    print("[INFO] out_dir:", out_dir)

    rows = []
    for i, p in enumerate(files, 1):
        row = clean_one(p, out_dir / p.name)
        rows.append(row)

        if i <= 5 or i % 10 == 0:
            print(
                f"[{i:03d}/{len(files)}] {p.stem} "
                f"D {row['D_old']}->{row['D_new']} "
                f"valid={row['valid_mean']:.3f} "
                f"mean={row['mean']:.3f} std={row['std']:.3f} "
                f"min={row['min']:.3f} max={row['max']:.3f}"
            )

    summary = pd.DataFrame(rows)
    out_csv = out_dir.parent / "build_maskshape_v1_clean_summary.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("[SAVE]", out_csv)
    print("[DONE] maskshape v1 clean finished.")


if __name__ == "__main__":
    main()

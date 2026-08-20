import os
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))
ASSET = ROOT / "data/cataract101_m10_a4_geo_assets_sam2/npz"
GT = ROOT / "data/cataract101_m10_a0_rgb2048/groundTruth"
OUT = ROOT / "artifacts/m10_a4_traj_analysis"
OUT.mkdir(parents=True, exist_ok=True)

DIM_NAMES = {
    0: "center_vx_norm",
    1: "center_vy_norm",
    2: "center_speed_norm",
    3: "rel_dist_delta",
    4: "rel_dx_delta",
    5: "rel_dy_delta",
    6: "presence",
    7: "valid_transition",
}

def load_gt_labels(p):
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def boundary_mask(labels, radius=10):
    T = len(labels)
    b = np.zeros(T, dtype=bool)
    for i in range(1, T):
        if labels[i] != labels[i - 1]:
            s = max(0, i - radius)
            e = min(T, i + radius + 1)
            b[s:e] = True
    return b

rows = []

for ap in sorted(ASSET.glob("case_*.npz")):
    case = ap.stem
    gt_path = GT / f"{case}.txt"
    if not gt_path.exists():
        continue

    z = np.load(ap)
    traj = z["traj"]                 # (2,8,T)
    geo = z["geo"]                   # (2,7,T), geo[:,2] = dist
    presence = z["presence_mask"]    # (2,T)
    labels = load_gt_labels(gt_path)
    T = min(traj.shape[-1], len(labels))

    traj = traj[:, :, :T]
    geo = geo[:, :, :T]
    presence = presence[:, :T]
    labels = labels[:T]

    bmask = boundary_mask(labels, radius=10)

    for tool in range(2):
        pm = presence[tool] > 0.5
        for dim in range(8):
            x = traj[tool, dim].astype(np.float32)
            valid = np.isfinite(x) & pm

            if valid.sum() < 10:
                continue

            xv = x[valid]
            absx = np.abs(xv)

            p95 = float(np.percentile(absx, 95))
            p99 = float(np.percentile(absx, 99))
            mx = float(np.max(absx))
            mean = float(np.mean(xv))
            std = float(np.std(xv))

            # spike ratio: values larger than 3 * p95 among valid frames
            thr = max(p95 * 3.0, 1e-6)
            spike_ratio = float((absx > thr).mean())

            # phase separation: between-phase variance / total variance
            phase_vals = {}
            for lab in sorted(set(labels)):
                mask = np.array([l == lab for l in labels]) & valid
                if mask.sum() >= 10:
                    phase_vals[lab] = float(np.mean(x[mask]))

            if len(phase_vals) >= 2 and std > 1e-6:
                phase_means = np.array(list(phase_vals.values()), dtype=np.float32)
                phase_sep = float(np.var(phase_means) / (std ** 2 + 1e-6))
            else:
                phase_sep = 0.0

            # boundary sensitivity
            boundary_valid = valid & bmask
            non_boundary_valid = valid & (~bmask)
            if boundary_valid.sum() >= 10 and non_boundary_valid.sum() >= 10:
                boundary_score = float(
                    np.mean(np.abs(x[boundary_valid])) /
                    (np.mean(np.abs(x[non_boundary_valid])) + 1e-6)
                )
            else:
                boundary_score = 0.0

            rows.append({
                "case": case,
                "tool": tool,
                "dim": dim,
                "dim_name": DIM_NAMES[dim],
                "valid_n": int(valid.sum()),
                "mean": mean,
                "std": std,
                "abs_p95": p95,
                "abs_p99": p99,
                "abs_max": mx,
                "spike_ratio": spike_ratio,
                "phase_sep": phase_sep,
                "boundary_score": boundary_score,
            })

df = pd.DataFrame(rows)
df.to_csv(OUT / "traj_dim_stats_per_case.csv", index=False, encoding="utf-8-sig")

summary = (
    df.groupby(["dim", "dim_name"])
      .agg({
          "valid_n": "sum",
          "std": "mean",
          "abs_p95": "mean",
          "abs_p99": "mean",
          "abs_max": "mean",
          "spike_ratio": "mean",
          "phase_sep": "mean",
          "boundary_score": "mean",
      })
      .reset_index()
)

summary["importance_score"] = (
    summary["phase_sep"] * 0.45
    + summary["boundary_score"] * 0.35
    - summary["spike_ratio"] * 0.20
)

summary = summary.sort_values("importance_score", ascending=False)
summary.to_csv(OUT / "traj_dim_global_summary.csv", index=False, encoding="utf-8-sig")

print("Saved:")
print(OUT / "traj_dim_stats_per_case.csv")
print(OUT / "traj_dim_global_summary.csv")
print()
print(summary.to_string(index=False))

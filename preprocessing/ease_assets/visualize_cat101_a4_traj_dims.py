import os
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))
ASSET = ROOT / "data/cataract101_m10_a4_geo_assets_sam2/npz"
GT = ROOT / "data/cataract101_m10_a0_rgb2048/groundTruth"
OUT = ROOT / "artifacts/m10_a4_traj_analysis"

DIM_NAMES = [
    "center_vx_norm",
    "center_vy_norm",
    "center_speed_norm",
    "rel_dist_delta",
    "rel_dx_delta",
    "rel_dy_delta",
    "presence",
    "valid_transition",
]

def load_labels(case):
    p = GT / f"{case}.txt"
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def phase_boundaries(labels):
    b = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            b.append(i)
    return b

def add_boundaries(ax, bounds):
    for b in bounds:
        ax.axvline(b, color="k", alpha=0.15, linewidth=0.8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default="case_269")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--smooth", type=int, default=1)
    args = ap.parse_args()

    case = args.case.replace(".npz", "").replace(".txt", "")
    z = np.load(ASSET / f"{case}.npz")
    labels = load_labels(case)

    traj = z["traj"]                 # (2,8,T)
    geo = z["geo"]                   # (2,7,T)
    presence = z["presence_mask"]    # (2,T)

    T = min(traj.shape[-1], len(labels))
    start = max(0, args.start)
    end = T if args.end < 0 else min(T, args.end)

    xs = np.arange(start, end)
    bounds = [b for b in phase_boundaries(labels[:T]) if start <= b < end]

    out_dir = OUT / "traj_dim_visuals" / case
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(11, 1, figsize=(18, 24), sharex=True)

    # phase label text blocks
    ax = axes[0]
    ax.set_title(f"{case} phase timeline")
    ax.plot(xs, np.zeros_like(xs), alpha=0)
    add_boundaries(ax, bounds)

    # write phase names at segment centers
    seg_start = start
    current = labels[start]
    for i in range(start + 1, end):
        if labels[i] != current:
            center = (seg_start + i) // 2
            ax.text(center, 0, current, rotation=30, fontsize=8, ha="center", va="center")
            seg_start = i
            current = labels[i]
    center = (seg_start + end - 1) // 2
    ax.text(center, 0, current, rotation=30, fontsize=8, ha="center", va="center")
    ax.set_yticks([])

    # distance
    ax = axes[1]
    ax.set_title("geo distance / eye_radius")
    for tool in range(2):
        y = geo[tool, 2, start:end]
        y = np.where(presence[tool, start:end] > 0.5, y, np.nan)
        ax.plot(xs, y, label=f"inst{tool} dist/r")
    add_boundaries(ax, bounds)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    # traj dims
    for dim in range(8):
        ax = axes[2 + dim]
        ax.set_title(f"traj[{dim}] {DIM_NAMES[dim]}")
        for tool in range(2):
            y = traj[tool, dim, start:end]
            if dim < 6:
                y = np.where(presence[tool, start:end] > 0.5, y, np.nan)
            ax.plot(xs, y, label=f"inst{tool}", linewidth=1)
        add_boundaries(ax, bounds)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("local frame index")

    plt.tight_layout()
    out_path = out_dir / f"{case}_traj_dims_{start}_{end}.png"
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

    print("[OK]", out_path)

if __name__ == "__main__":
    main()

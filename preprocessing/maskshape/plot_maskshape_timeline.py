# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE_SHORT = {
    "incision": "Incision",
    "viscous_agent_injection": "Viscous Inj.",
    "rhexis": "Rhexis",
    "hydrodissection": "Hydrodissection",
    "phacoemulsificiation": "Phaco",
    "phacoemulsification": "Phaco",
    "irrigation_and_aspiration": "I&A",
    "capsule_polishing": "Capsule Polish",
    "lens_implant_setting_up": "Lens Implant",
    "viscous_agent_removal": "Viscous Removal",
    "tonifying_and_antibiotics": "Antibiotics",
}


def short_phase(x):
    return PHASE_SHORT.get(str(x), str(x))


def load_gt(gt_dir, case):
    p = Path(gt_dir) / f"{case}.txt"
    arr = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                arr.append(s)
    return np.asarray(arr, dtype=object)


def phase_to_ids(gt):
    names = []
    mp = {}
    ids = []
    for x in gt:
        sx = short_phase(x)
        if sx not in mp:
            mp[sx] = len(mp)
            names.append(sx)
        ids.append(mp[sx])
    return np.asarray(ids, dtype=np.int32), names


def get_idx(names, target):
    for i, n in enumerate(names):
        if str(n) == target:
            return i
    raise KeyError(target)


def plot_case(case, mask_dir, gt_dir, out_dir, stride=10):
    p = Path(mask_dir) / f"{case}.npz"
    if not p.exists():
        print("[WARN] missing", p)
        return

    z = np.load(p, allow_pickle=True)
    x = z["mask_shape"].astype(np.float32)       # (2,D,T)
    valid = z["mask_shape_valid"].astype(np.float32)
    names = [str(n) for n in z["mask_shape_names"]]
    gt = load_gt(gt_dir, case)

    T = min(x.shape[-1], valid.shape[-1], len(gt))
    x = x[..., :T]
    valid = valid[..., :T]
    gt = gt[:T]

    ids = np.arange(0, T, stride)
    phase_ids, phase_names = phase_to_ids(gt)

    idx_area = get_idx(names, "inst_area_ratio")
    idx_union_area = get_idx(names, "union_area_ratio")
    idx_score = get_idx(names, "inst_score")
    idx_center_dist = get_idx(names, "center_dist_norm")
    idx_union_dist = get_idx(names, "union_dist_norm")
    idx_compact = get_idx(names, "union_compactness")

    fig, axes = plt.subplots(8, 1, figsize=(18, 14), sharex=True)

    axes[0].plot(ids, phase_ids[ids], linewidth=1.2)
    axes[0].set_ylabel("phase")
    axes[0].set_title(f"{case}: A5 mask-shape v1-safe timeline", fontsize=15, fontweight="bold")

    for tr in range(2):
        axes[1].plot(ids, x[tr, idx_area, ids], linewidth=0.9, label=f"track{tr}")
    axes[1].set_ylabel("inst area")
    axes[1].legend(loc="upper right")

    axes[2].plot(ids, x[0, idx_union_area, ids], linewidth=0.9, label="union area")
    axes[2].set_ylabel("union area")
    axes[2].legend(loc="upper right")

    for tr in range(2):
        axes[3].plot(ids, x[tr, idx_center_dist, ids], linewidth=0.9, label=f"track{tr}")
    axes[3].set_ylabel("center dist")
    axes[3].legend(loc="upper right")

    axes[4].plot(ids, x[0, idx_union_dist, ids], linewidth=0.9, label="union dist")
    axes[4].set_ylabel("union dist")
    axes[4].legend(loc="upper right")

    axes[5].plot(ids, x[0, idx_compact, ids], linewidth=0.9, label="compactness")
    axes[5].set_ylabel("compact")
    axes[5].legend(loc="upper right")

    for tr in range(2):
        axes[6].plot(ids, x[tr, idx_score, ids], linewidth=0.9, label=f"track{tr}")
    axes[6].set_ylabel("inst score")
    axes[6].legend(loc="upper right")

    for tr in range(2):
        axes[7].plot(ids, valid[tr, ids], linewidth=0.9, label=f"track{tr}")
    axes[7].set_ylabel("valid")
    axes[7].set_xlabel("frame index")
    axes[7].legend(loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    out = Path(out_dir) / f"maskshape_timeline_{case}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[SAVE]", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", default="data/cataract101_m10_a5_maskshape_assets_v1_safe/npz")
    ap.add_argument("--gt_dir", default="data/cataract101_m10_a5e_base_from_a4v1d_fixaux/groundTruth")
    ap.add_argument("--out_dir", default="artifacts/m10_maskshape_viz")
    ap.add_argument("--cases", default="case_738,case_269,case_270,case_828,case_825")
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for case in [c.strip() for c in args.cases.split(",") if c.strip()]:
        plot_case(case, args.mask_dir, args.gt_dir, args.out_dir, stride=args.stride)

    print("[DONE] mask-shape timeline visualization finished.")


if __name__ == "__main__":
    main()

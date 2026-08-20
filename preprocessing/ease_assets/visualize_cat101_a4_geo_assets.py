import os
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))
CAT = Path(os.environ.get("CATARACT101_ROOT", "/path/to/cataract-101"))
ASSET_DIR = ROOT / "data/cataract101_m10_a4_geo_assets_sam2/npz"


def norm_to_pixel_xy(x, y, W, H):
    x = float(x)
    y = float(y)

    # If coordinates look normalized, convert to pixels.
    if -2.0 <= x <= 2.0 and -2.0 <= y <= 2.0:
        return int(round(x * W)), int(round(y * H))

    return int(round(x)), int(round(y))


def norm_to_pixel_radius(r, W, H):
    r = float(r)

    if 0 <= r <= 2.0:
        return int(round(r * min(W, H)))

    return int(round(r))


def clamp_pt(x, y, W, H):
    x = max(0, min(W - 1, int(x)))
    y = max(0, min(H - 1, int(y)))
    return x, y


def draw_soft_anchors(rgb, asset, t, future_gap=5):
    out = rgb.copy()
    H, W = out.shape[:2]

    eye_center = asset["eye_center"]      # (2,T)
    eye_radius = asset["eye_radius"]      # (T,)
    inst_tips = asset["inst_tips"]        # (2,2,T)
    presence = asset["presence_mask"]     # (2,T)

    ex, ey = norm_to_pixel_xy(eye_center[0, t], eye_center[1, t], W, H)
    er = norm_to_pixel_radius(eye_radius[t], W, H)
    ex, ey = clamp_pt(ex, ey, W, H)

    # eye circle and center
    cv2.circle(out, (ex, ey), max(er, 2), (0, 255, 0), 2)
    cv2.circle(out, (ex, ey), 5, (0, 255, 0), -1)
    cv2.putText(out, f"Eye r={float(eye_radius[t]):.3f}", (ex + 6, ey - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    colors = [(255, 80, 40), (255, 190, 0)]

    active_dist = []
    for j in range(2):
        if presence[j, t] <= 0.5:
            continue

        tx, ty = norm_to_pixel_xy(inst_tips[j, 0, t], inst_tips[j, 1, t], W, H)
        tx, ty = clamp_pt(tx, ty, W, H)

        color = colors[j]
        cv2.circle(out, (tx, ty), 6, color, -1)
        cv2.line(out, (ex, ey), (tx, ty), color, 2)
        cv2.putText(out, f"Inst{j}", (tx + 5, ty - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # motion arrow from current to future
        t2 = min(t + future_gap, inst_tips.shape[-1] - 1)
        if presence[j, t2] > 0.5:
            fx, fy = norm_to_pixel_xy(inst_tips[j, 0, t2], inst_tips[j, 1, t2], W, H)
            fx, fy = clamp_pt(fx, fy, W, H)
            cv2.arrowedLine(out, (tx, ty), (fx, fy), color, 2, tipLength=0.35)

        d = ((tx - ex) ** 2 + (ty - ey) ** 2) ** 0.5
        active_dist.append((j, d))

    if active_dist:
        active_dist.sort(key=lambda x: x[1])
        j, d = active_dist[0]
        cv2.putText(out, f"Active inst={j}, dist={d:.1f}px", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def get_rgb(video_id, frame_id):
    video_path = CAT / "videos" / f"case_{video_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"cannot read frame {frame_id}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
    return rgb


def plot_geo_curves(ax, asset, t, window=120):
    geo = asset["geo"]                    # (2,7,T)
    presence = asset["presence_mask"]     # (2,T)
    T = geo.shape[-1]

    s = max(0, t - window)
    e = min(T, t + window + 1)
    xs = np.arange(s, e)

    for j in range(2):
        dist = geo[j, 2, s:e]
        dist = np.where(presence[j, s:e] > 0.5, dist, np.nan)
        ax.plot(xs, dist, label=f"inst{j} dist/r")

    ax.axvline(t, linestyle="--")
    ax.set_title("Relative distance around current frame")
    ax.set_xlabel("local frame index")
    ax.set_ylabel("distance / eye_radius")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def save_one(video_id, local_idx, out_dir, future_gap=5):
    asset_path = ASSET_DIR / f"case_{video_id}.npz"
    asset = np.load(asset_path)

    frame_ids = asset["frame_ids"]
    T = len(frame_ids)

    if local_idx < 0 or local_idx >= T:
        raise ValueError(f"local_idx={local_idx}, T={T}")

    frame_id = int(frame_ids[local_idx])
    rgb = get_rgb(video_id, frame_id)

    soft = draw_soft_anchors(rgb, asset, local_idx, future_gap=future_gap)

    geo = asset["geo"]
    aux = asset["aux"]
    presence = asset["presence_mask"]
    traj = asset["traj"]

    fig = plt.figure(figsize=(15, 8))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title("RGB")
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(soft)
    ax2.set_title("Soft anchors: eye / instrument / motion")
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    plot_geo_curves(ax3, asset, local_idx)

    ax4 = plt.subplot(2, 3, 4)
    ax4.axis("off")
    txt = []
    txt.append(f"case_{video_id}")
    txt.append(f"local_idx={local_idx}")
    txt.append(f"frame_id={frame_id}")
    txt.append("")
    txt.append(f"eye_valid={aux[0, local_idx]:.3f}")
    txt.append(f"inst_any={aux[1, local_idx]:.3f}")
    txt.append(f"inst_count_norm={aux[2, local_idx]:.3f}")
    txt.append(f"union_area/eye_area={aux[3, local_idx]:.3f}")
    txt.append("")
    for j in range(2):
        txt.append(f"inst{j} presence={presence[j, local_idx]:.1f}")
        txt.append(f"  dx={geo[j,0,local_idx]:.3f}, dy={geo[j,1,local_idx]:.3f}")
        txt.append(f"  dist={geo[j,2,local_idx]:.3f}")
        txt.append(f"  area_ratio={geo[j,5,local_idx]:.3f}")
        txt.append(f"  speed={traj[j,2,local_idx]:.4f}")
    ax4.text(0.02, 0.98, "\n".join(txt), va="top", ha="left", fontsize=10, family="monospace")
    ax4.set_title("A4 geometry values")

    ax5 = plt.subplot(2, 3, 5)
    for j in range(2):
        speed = traj[j, 2]
        s = max(0, local_idx - 120)
        e = min(T, local_idx + 121)
        xs = np.arange(s, e)
        vals = np.where(presence[j, s:e] > 0.5, speed[s:e], np.nan)
        ax5.plot(xs, vals, label=f"inst{j} speed")
    ax5.axvline(local_idx, linestyle="--")
    ax5.set_title("Trajectory speed around current frame")
    ax5.grid(True, alpha=0.25)
    ax5.legend(fontsize=8)

    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(soft)
    ax6.set_title("Combined anchor view")
    ax6.axis("off")

    plt.suptitle(f"A4 Geo Asset Visualization | case_{video_id} | frame={frame_id}", fontsize=14)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"case_{video_id}_idx_{local_idx:06d}_frame_{frame_id:06d}_a4_geo.png"
    plt.savefig(out_path, dpi=160)
    plt.close(fig)

    print("[OK]", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", type=int, default=269)
    ap.add_argument("--indices", type=str, default="0,1200,2400,3600,4800")
    ap.add_argument("--out_dir", type=str, default="artifacts/m10_a4_geo_asset_vis")
    ap.add_argument("--future_gap", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / f"case_{args.video_id}"
    indices = [int(x) for x in args.indices.split(",") if x.strip()]

    for idx in indices:
        save_one(args.video_id, idx, out_dir, future_gap=args.future_gap)

    print("[DONE] saved to:", out_dir)


if __name__ == "__main__":
    main()

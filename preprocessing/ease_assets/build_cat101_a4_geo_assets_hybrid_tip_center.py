import os
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))

# v1c-cons: precise tip-track
TIP_DIR = ROOT / "data/cataract101_m10_a4_geo_assets_sam2_yolo_tiptrack_cons/npz"

# v0: original center-proxy geometry asset
CENTER_DIR = ROOT / "data/cataract101_m10_a4_geo_assets_sam2/npz"

OUT_ROOT = ROOT / "data/cataract101_m10_a4_geo_assets_hybrid_tip_center"
OUT_NPZ = OUT_ROOT / "npz"
OUT_META = OUT_ROOT / "meta"

GEO_DXY_CLIP = 10.0
GEO_DIST_CLIP = 10.0
AREA_RATIO_CLIP = 5.0
TRAJ_MOTION_CLIP = 20.0


def clip_arr(x, lo, hi):
    return np.clip(np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=hi, neginf=lo), lo, hi)


def get_tips_from_asset(z):
    """
    Prefer inst_tips. If absent, fallback to geo tip_x/tip_y.
    Return shape (2,2,T).
    """
    if "inst_tips" in z.files:
        tips = z["inst_tips"].astype(np.float32)
        if tips.shape[1] == 2:
            return tips

    geo = z["geo"].astype(np.float32)
    T = geo.shape[-1]
    tips = np.zeros((2, 2, T), dtype=np.float32)
    tips[:, 0, :] = geo[:, 3, :]
    tips[:, 1, :] = geo[:, 4, :]
    return tips


def get_presence_from_asset(z):
    if "presence_mask" in z.files:
        return z["presence_mask"].astype(np.float32)
    geo = z["geo"].astype(np.float32)
    return (geo[:, 6, :] > 0.5).astype(np.float32)


def build_one(case_name, force=False):
    out_p = OUT_NPZ / f"{case_name}.npz"
    out_s = OUT_META / f"{case_name}_summary.json"

    if out_p.exists() and out_s.exists() and not force:
        return {"case": case_name, "status": "skip"}

    tip_p = TIP_DIR / f"{case_name}.npz"
    center_p = CENTER_DIR / f"{case_name}.npz"

    if not tip_p.exists():
        return {"case": case_name, "status": "missing_tip", "path": str(tip_p)}
    if not center_p.exists():
        return {"case": case_name, "status": "missing_center", "path": str(center_p)}

    tip_z = np.load(tip_p, allow_pickle=True)
    cen_z = np.load(center_p, allow_pickle=True)

    # Basic reference
    frame_ids = tip_z["frame_ids"] if "frame_ids" in tip_z.files else cen_z["frame_ids"]
    T = len(frame_ids)

    tip_geo = tip_z["geo"].astype(np.float32)
    cen_geo = cen_z["geo"].astype(np.float32)

    tip_presence = get_presence_from_asset(tip_z)[:, :T]
    cen_presence = get_presence_from_asset(cen_z)[:, :T]

    tip_tips = get_tips_from_asset(tip_z)[:, :, :T]
    cen_tips = get_tips_from_asset(cen_z)[:, :, :T]

    eye_center = tip_z["eye_center"].astype(np.float32) if "eye_center" in tip_z.files else cen_z["eye_center"].astype(np.float32)
    eye_radius = tip_z["eye_radius"].astype(np.float32) if "eye_radius" in tip_z.files else cen_z["eye_radius"].astype(np.float32)

    # eye_center should be (2,T)
    if eye_center.shape[0] != 2 and eye_center.shape[-1] == 2:
        eye_center = eye_center.T

    eye_center = eye_center[:, :T]
    eye_radius = eye_radius[:T]

    # Hybrid tips and source
    hybrid_tips = np.zeros((2, 2, T), dtype=np.float32)
    hybrid_presence = np.zeros((2, T), dtype=np.float32)
    source_mask = np.zeros((2, T), dtype=np.int8)  # 0 none, 1 tip-track, 2 center-fallback

    # First fill tip-track
    for tr in range(2):
        m = tip_presence[tr] > 0.5
        hybrid_tips[tr, :, m] = tip_tips[tr, :, m]
        hybrid_presence[tr, m] = 1.0
        source_mask[tr, m] = 1

    # Then fill center-proxy only when missing
    for tr in range(2):
        m = (hybrid_presence[tr] <= 0.5) & (cen_presence[tr] > 0.5)
        hybrid_tips[tr, :, m] = cen_tips[tr, :, m]
        hybrid_presence[tr, m] = 1.0
        source_mask[tr, m] = 2

    # Recompute geo/traj from hybrid tips
    geo = np.zeros((2, 7, T), dtype=np.float32)
    traj = np.zeros((2, 8, T), dtype=np.float32)

    ex = eye_center[0]
    ey = eye_center[1]
    r = np.maximum(eye_radius.astype(np.float32), 1e-6)

    # area_ratio: use source-specific geo[:,5] when available
    area_ratio = np.zeros((2, T), dtype=np.float32)
    for tr in range(2):
        tip_m = source_mask[tr] == 1
        cen_m = source_mask[tr] == 2
        if tip_geo.shape[1] > 5:
            area_ratio[tr, tip_m] = tip_geo[tr, 5, :T][tip_m]
        if cen_geo.shape[1] > 5:
            area_ratio[tr, cen_m] = cen_geo[tr, 5, :T][cen_m]

    for tr in range(2):
        px = hybrid_tips[tr, 0]
        py = hybrid_tips[tr, 1]
        pres = hybrid_presence[tr] > 0.5

        dx = (px - ex) / r
        dy = (py - ey) / r
        dist = np.sqrt(dx * dx + dy * dy)

        geo[tr, 0] = np.where(pres, dx, 0.0)
        geo[tr, 1] = np.where(pres, dy, 0.0)
        geo[tr, 2] = np.where(pres, dist, 0.0)
        geo[tr, 3] = np.where(pres, px, 0.0)
        geo[tr, 4] = np.where(pres, py, 0.0)
        geo[tr, 5] = np.where(pres, area_ratio[tr], 0.0)
        geo[tr, 6] = pres.astype(np.float32)

        valid_transition = np.zeros(T, dtype=bool)
        valid_transition[1:] = (
            pres[1:]
            & pres[:-1]
            & (source_mask[tr, 1:] == source_mask[tr, :-1])
        )

        vx = np.zeros(T, dtype=np.float32)
        vy = np.zeros(T, dtype=np.float32)
        vx[1:] = px[1:] - px[:-1]
        vy[1:] = py[1:] - py[:-1]

        vx_norm = vx / r
        vy_norm = vy / r
        speed = np.sqrt(vx_norm * vx_norm + vy_norm * vy_norm)

        ddist = np.zeros(T, dtype=np.float32)
        ddx = np.zeros(T, dtype=np.float32)
        ddy = np.zeros(T, dtype=np.float32)
        ddist[1:] = dist[1:] - dist[:-1]
        ddx[1:] = dx[1:] - dx[:-1]
        ddy[1:] = dy[1:] - dy[:-1]

        traj[tr, 0] = np.where(valid_transition, vx_norm, 0.0)
        traj[tr, 1] = np.where(valid_transition, vy_norm, 0.0)
        traj[tr, 2] = np.where(valid_transition, speed, 0.0)
        traj[tr, 3] = np.where(valid_transition, ddist, 0.0)
        traj[tr, 4] = np.where(valid_transition, ddx, 0.0)
        traj[tr, 5] = np.where(valid_transition, ddy, 0.0)
        traj[tr, 6] = pres.astype(np.float32)
        traj[tr, 7] = valid_transition.astype(np.float32)

    # light-clip
    geo[:, 0:2] = clip_arr(geo[:, 0:2], -GEO_DXY_CLIP, GEO_DXY_CLIP)
    geo[:, 2] = clip_arr(geo[:, 2], 0.0, GEO_DIST_CLIP)
    geo[:, 5] = clip_arr(geo[:, 5], 0.0, AREA_RATIO_CLIP)

    traj[:, 0:2] = clip_arr(traj[:, 0:2], -TRAJ_MOTION_CLIP, TRAJ_MOTION_CLIP)
    traj[:, 2] = clip_arr(traj[:, 2], 0.0, TRAJ_MOTION_CLIP)
    traj[:, 3:6] = clip_arr(traj[:, 3:6], -TRAJ_MOTION_CLIP, TRAJ_MOTION_CLIP)
    traj[:, 6] = (traj[:, 6] > 0.5).astype(np.float32)
    traj[:, 7] = (traj[:, 7] > 0.5).astype(np.float32)

    save_dict = {}

    # preserve useful keys from tip asset
    for k in tip_z.files:
        if k not in ["geo", "traj", "inst_tips", "presence_mask"]:
            save_dict[k] = tip_z[k]

    save_dict.update({
        "frame_ids": frame_ids,
        "eye_center": eye_center.astype(np.float32),
        "eye_radius": eye_radius.astype(np.float32),
        "inst_tips": hybrid_tips.astype(np.float32),
        "presence_mask": hybrid_presence.astype(np.float32),
        "geo": geo.astype(np.float32),
        "traj": traj.astype(np.float32),
        "source_mask": source_mask.astype(np.int8),
    })

    OUT_NPZ.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)
    np.savez(out_p, **save_dict)

    pos_dist = geo[:, 2][geo[:, 2] > 0]
    pos_speed = traj[:, 2][traj[:, 2] > 0]

    summary = {
        "case": case_name,
        "status": "ok",
        "T": int(T),
        "tip_presence_mean": float(tip_presence.mean()),
        "center_presence_mean": float(cen_presence.mean()),
        "hybrid_presence_mean": float(hybrid_presence.mean()),
        "tip_source_ratio": float((source_mask == 1).mean()),
        "center_fallback_ratio": float((source_mask == 2).mean()),
        "cand_count_ge2_ratio": float((hybrid_presence.sum(axis=0) >= 2).mean()),
        "geo_dist_p95": float(np.percentile(pos_dist, 95)) if len(pos_dist) else 0.0,
        "geo_dist_max": float(np.max(pos_dist)) if len(pos_dist) else 0.0,
        "traj_speed_p95": float(np.percentile(pos_speed, 95)) if len(pos_speed) else 0.0,
        "traj_speed_max": float(np.max(pos_speed)) if len(pos_speed) else 0.0,
    }

    out_s.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tip_cases = {p.stem for p in TIP_DIR.glob("case_*.npz")}
    center_cases = {p.stem for p in CENTER_DIR.glob("case_*.npz")}
    cases = sorted(tip_cases & center_cases, key=lambda x: int(x.replace("case_", "")))

    if args.limit > 0:
        cases = cases[:args.limit]

    print("=" * 80)
    print("[A4-v1d hybrid tip-center asset]")
    print("TIP_DIR:", TIP_DIR)
    print("CENTER_DIR:", CENTER_DIR)
    print("OUT_NPZ:", OUT_NPZ)
    print("cases:", len(cases), cases[:10])
    print("=" * 80)

    all_s = []
    for i, c in enumerate(cases, 1):
        s = build_one(c, force=args.force)
        all_s.append(s)
        print(f"[{i}/{len(cases)}]", json.dumps(s, ensure_ascii=False), flush=True)

    ok = [s for s in all_s if s.get("status") in ["ok", "skip"]]
    print("[DONE] ok/total:", len(ok), "/", len(all_s))


if __name__ == "__main__":
    main()

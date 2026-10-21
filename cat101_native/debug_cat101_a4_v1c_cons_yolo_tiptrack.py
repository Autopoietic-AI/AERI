# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("/root/autodl-tmp/MS-TCN2")
CAT = Path("/root/autodl-tmp/101data/cataract-101")
SAM2 = ROOT / "data/cataract101_m10_a2c_sam2_masks_full/npz"
YOLO = ROOT / "data/cataract101_m10_yolo_b2_multiinst_assets/npz"
GT = ROOT / "data/cataract101_m10_a0_rgb2048/groundTruth"
OUT_ROOT = ROOT / "artifacts/m10_a4_tiptrack_v1c_cons_debug"


def log(x):
    print(x, flush=True)


def unpack_mask(packed, h, w):
    bits = np.unpackbits(packed, bitorder="little")[: h * w]
    return bits.reshape(h, w).astype(np.uint8)


def norm_eye_center(x, w, h):
    x = np.asarray(x, dtype=np.float32).copy()
    if np.nanmax(x[:, 0]) > 2 or np.nanmax(x[:, 1]) > 2:
        x[:, 0] /= max(w - 1, 1)
        x[:, 1] /= max(h - 1, 1)
    return x


def norm_eye_area(a, h, w):
    a = np.asarray(a, dtype=np.float32).copy()
    med = np.nanmedian(a[np.isfinite(a)]) if np.isfinite(a).any() else 0
    if med > 1.5:
        a = a / float(h * w)
    return a


def norm_boxes(boxes, w, h):
    b = boxes.astype(np.float32).copy()
    if np.nanmax(b[..., [0, 2]]) > 2 or np.nanmax(b[..., [1, 3]]) > 2:
        b[..., [0, 2]] /= max(w - 1, 1)
        b[..., [1, 3]] /= max(h - 1, 1)
    b[..., [0, 2]] = np.clip(b[..., [0, 2]], 0, 1)
    b[..., [1, 3]] = np.clip(b[..., [1, 3]], 0, 1)
    return b


def box_area(b):
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def inter_area(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a, b):
    inter = inter_area(a, b)
    return inter / (box_area(a) + box_area(b) - inter + 1e-9)


def containment(a, b):
    inter = inter_area(a, b)
    return inter / (min(box_area(a), box_area(b)) + 1e-9)


def center_dist_px(a, b, w, h):
    ca = np.array([(a[0] + a[2]) * 0.5 * w, (a[1] + a[3]) * 0.5 * h], dtype=np.float32)
    cb = np.array([(b[0] + b[2]) * 0.5 * w, (b[1] + b[3]) * 0.5 * h], dtype=np.float32)
    return float(np.linalg.norm(ca - cb))


def merge_boxes(a, b):
    return np.array([
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ], dtype=np.float32)


def expand_box(b, expand=0.08):
    x1, y1, x2, y2 = [float(v) for v in b]
    bw = x2 - x1
    bh = y2 - y1
    return np.array([
        max(0.0, x1 - expand * bw),
        max(0.0, y1 - expand * bh),
        min(1.0, x2 + expand * bw),
        min(1.0, y2 + expand * bh),
    ], dtype=np.float32)


def largest_component(mask, min_area=20):
    if int(mask.sum()) < min_area:
        return None

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return None

    best_lab, best_area = None, 0
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area > best_area:
            best_lab, best_area = lab, area

    if best_area < min_area:
        return None

    return (labels == best_lab).astype(np.uint8)


def tip_from_region(region, eye_xy, min_area=20):
    h, w = region.shape
    ys, xs = np.where(region > 0)
    if len(xs) < min_area:
        return None

    xy = np.stack([xs, ys], axis=1).astype(np.float32)
    center_px = xy.mean(axis=0)
    eye_px = np.array([eye_xy[0] * (w - 1), eye_xy[1] * (h - 1)], dtype=np.float32)

    # closest-to-eye, average closest 1%
    d = np.linalg.norm(xy - eye_px[None, :], axis=1)
    k = max(3, min(50, int(len(d) * 0.01)))
    closest_tip = xy[np.argsort(d)[:k]].mean(axis=0)

    # PCA long-axis tip
    pca_tip = closest_tip.copy()
    pca_ok = False
    elongation = 1.0

    if len(xy) >= 10:
        xyc = xy - center_px[None, :]
        cov = np.cov(xyc.T)
        if np.isfinite(cov).all():
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            if vals[0] > 1e-6:
                elongation = float(vals[0] / max(vals[1], 1e-6))
                v = vecs[:, 0]
                proj = xyc @ v
                end1 = center_px + v * proj.min()
                end2 = center_px + v * proj.max()
                pca_tip = end1 if np.linalg.norm(end1 - eye_px) <= np.linalg.norm(end2 - eye_px) else end2
                pca_ok = elongation >= 2.0

    tip_px = pca_tip if pca_ok else closest_tip

    for pt in [center_px, closest_tip, pca_tip, tip_px]:
        pt[0] = np.clip(pt[0], 0, w - 1)
        pt[1] = np.clip(pt[1], 0, h - 1)

    return {
        "area": int(len(xs)),
        "center": np.array([center_px[0] / (w - 1), center_px[1] / (h - 1)], dtype=np.float32),
        "tip": np.array([tip_px[0] / (w - 1), tip_px[1] / (h - 1)], dtype=np.float32),
        "closest_tip": np.array([closest_tip[0] / (w - 1), closest_tip[1] / (h - 1)], dtype=np.float32),
        "pca_tip": np.array([pca_tip[0] / (w - 1), pca_tip[1] / (h - 1)], dtype=np.float32),
        "pca_ok": bool(pca_ok),
        "elongation": float(elongation),
    }


def region_from_box(union_mask, box, eye_xy, expand=0.08, min_area=20):
    h, w = union_mask.shape
    b = expand_box(box, expand=expand)

    x1 = int(round(b[0] * (w - 1)))
    y1 = int(round(b[1] * (h - 1)))
    x2 = int(round(b[2] * (w - 1)))
    y2 = int(round(b[3] * (h - 1)))

    region = np.zeros_like(union_mask, dtype=np.uint8)
    region[y1:y2 + 1, x1:x2 + 1] = union_mask[y1:y2 + 1, x1:x2 + 1]

    comp = largest_component(region, min_area=min_area)
    if comp is None:
        return None

    return tip_from_region(comp, eye_xy, min_area=min_area)


def center_split_fallback(union_mask, centers_norm, eye_xy, min_area=20):
    """
    Split union mask by two YOLO box centers when YOLO duplicate leaves only one box.
    """
    h, w = union_mask.shape
    ys, xs = np.where(union_mask > 0)
    out = [None, None]

    if len(xs) < min_area:
        return out

    valid_centers = []
    valid_ids = []
    for i in range(2):
        cx, cy = centers_norm[i]
        if np.isfinite([cx, cy]).all() and -0.2 <= cx <= 1.2 and -0.2 <= cy <= 1.2:
            valid_ids.append(i)
            valid_centers.append([cx * (w - 1), cy * (h - 1)])

    if len(valid_ids) == 0:
        return out

    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    if len(valid_ids) == 1:
        comp = largest_component(union_mask, min_area=min_area)
        if comp is not None:
            out[valid_ids[0]] = tip_from_region(comp, eye_xy, min_area=min_area)
        return out

    centers = np.asarray(valid_centers, dtype=np.float32)
    dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    assign = np.argmin(dists, axis=1)

    for local_id, inst_id in enumerate(valid_ids):
        region = np.zeros_like(union_mask, dtype=np.uint8)
        sub = pts[assign == local_id].astype(np.int32)
        if len(sub) < min_area:
            continue
        region[sub[:, 1], sub[:, 0]] = 1
        comp = largest_component(region, min_area=min_area)
        if comp is not None:
            out[inst_id] = tip_from_region(comp, eye_xy, min_area=min_area)

    return out


def build_candidates(union_mask, boxes_t, conf_t, valid_t, eye_xy, w, h,
                     conf_th=0.30, box_expand=0.08, min_area=20):
    """
    Return candidates list length 2.
    Handles top-2 duplicate boxes by merging and fallback split.
    """
    raw = []
    for i in range(2):
        if valid_t[i] <= 0.5 or conf_t[i] < conf_th:
            continue
        b = boxes_t[i]
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        raw.append((i, b, float(conf_t[i])))

    # no valid YOLO boxes
    if len(raw) == 0:
        return [None, None], "none"

    # one valid YOLO box
    if len(raw) == 1:
        idx, b, conf = raw[0]
        cand = region_from_box(union_mask, b, eye_xy, expand=box_expand, min_area=min_area)
        out = [None, None]
        out[idx] = cand
        return out, "one_box"

    # two valid boxes
    i0, b0, c0 = raw[0]
    i1, b1, c1 = raw[1]

    v_iou = box_iou(b0, b1)
    v_contain = containment(b0, b1)
    cd = center_dist_px(b0, b1, w, h)

    # Conservative duplicate rule:
    # moderate duplicate: likely duplicated boxes, but may still contain two nearby tools
    # strong duplicate: very likely one tool represented by big+small boxes; do NOT force a second track
    moderate_duplicate = (v_iou > 0.50) or (v_contain > 0.75) or (v_contain > 0.60 and cd < 50.0)
    strong_duplicate = (v_contain > 0.90) or (v_contain > 0.75 and cd < 40.0)

    out = [None, None]

    if moderate_duplicate:
        merged = merge_boxes(b0, b1)
        main_idx = i0 if c0 >= c1 else i1
        cand_main = region_from_box(union_mask, merged, eye_xy, expand=box_expand, min_area=min_area)
        out[main_idx] = cand_main

        # If boxes are strongly contained, treat them as one instrument only.
        # This suppresses false two-track splitting for one tool.
        if strong_duplicate:
            return out, "strong_duplicate_merge_only"

        # Otherwise, try fallback to recover a second real instrument.
        centers_norm = np.zeros((2, 2), dtype=np.float32)
        for inst_i, box in [(i0, b0), (i1, b1)]:
            centers_norm[inst_i] = [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]

        fallback = center_split_fallback(union_mask, centers_norm, eye_xy, min_area=min_area)

        for j in range(2):
            if out[j] is None and fallback[j] is not None:
                out[j] = fallback[j]

        return out, "moderate_duplicate_merge_fallback"

    # not duplicate: use two boxes separately
    for idx, b, conf in raw:
        out[idx] = region_from_box(union_mask, b, eye_xy, expand=box_expand, min_area=min_area)

    return out, "two_boxes"


def track_cost(prev_tip, prev_center, cand):
    return float(np.linalg.norm(prev_tip - cand["tip"]) + 0.5 * np.linalg.norm(prev_center - cand["center"]))


def assign_tracks(cands, prev_tip, prev_center, prev_active):
    available = [i for i, c in enumerate(cands) if c is not None]
    assign = [None, None]

    if len(available) == 0:
        return assign

    active = [i for i in range(2) if prev_active[i]]

    if len(active) == 0:
        # initialize by x coordinate
        order = sorted(available, key=lambda i: cands[i]["tip"][0])
        for tr, ci in enumerate(order[:2]):
            assign[tr] = ci
        return assign

    if len(available) == 1:
        ci = available[0]
        best_tr, best_cost = active[0], 1e9
        for tr in active:
            c = track_cost(prev_tip[tr], prev_center[tr], cands[ci])
            if c < best_cost:
                best_cost = c
                best_tr = tr
        assign[best_tr] = ci
        return assign

    c0, c1 = available[:2]

    if len(active) == 1:
        tr = active[0]
        other = 1 - tr
        if track_cost(prev_tip[tr], prev_center[tr], cands[c0]) <= track_cost(prev_tip[tr], prev_center[tr], cands[c1]):
            assign[tr], assign[other] = c0, c1
        else:
            assign[tr], assign[other] = c1, c0
        return assign

    a = track_cost(prev_tip[0], prev_center[0], cands[c0]) + track_cost(prev_tip[1], prev_center[1], cands[c1])
    b = track_cost(prev_tip[0], prev_center[0], cands[c1]) + track_cost(prev_tip[1], prev_center[1], cands[c0])

    if a <= b:
        assign[0], assign[1] = c0, c1
    else:
        assign[0], assign[1] = c1, c0

    return assign


def build_tiptrack(video_id, out_dir, conf_th=0.30, box_expand=0.08, min_area=20):
    sp = SAM2 / f"case_{video_id}.npz"
    yp = YOLO / f"case_{video_id}.npz"

    s = np.load(sp)
    y = np.load(yp, allow_pickle=True)

    frame_ids = s["frame_ids"]
    h = int(s["mask_h"])
    w = int(s["mask_w"])
    T = len(frame_ids)

    eye_center = norm_eye_center(s["eye_center"], w, h)
    eye_area = norm_eye_area(s["eye_area"], h, w)
    eye_radius = np.sqrt(np.maximum(eye_area, 1e-8) / np.pi).astype(np.float32)

    boxes = norm_boxes(y["inst_xyxy_multi"], int(y["width"]), int(y["height"]))
    confs = y["inst_conf_multi"].astype(np.float32)
    valid = y["inst_valid_multi"].astype(np.float32)

    if len(boxes) != T:
        raise RuntimeError(f"T mismatch: sam2={T}, yolo={len(boxes)}")

    tips = np.zeros((2, 2, T), dtype=np.float32)
    centers = np.zeros((2, 2, T), dtype=np.float32)
    presence = np.zeros((2, T), dtype=np.float32)
    pca_ok = np.zeros((2, T), dtype=np.float32)
    area = np.zeros((2, T), dtype=np.float32)

    cand_count = np.zeros(T, dtype=np.int32)
    mode_count = {}

    prev_tip = np.zeros((2, 2), dtype=np.float32)
    prev_center = np.zeros((2, 2), dtype=np.float32)
    prev_active = np.zeros(2, dtype=bool)

    for t in range(T):
        union_mask = unpack_mask(s["inst_union_mask_packed"][t], h, w)
        cands, mode = build_candidates(
            union_mask, boxes[t], confs[t], valid[t], eye_center[t],
            w=w, h=h, conf_th=conf_th, box_expand=box_expand, min_area=min_area
        )

        mode_count[mode] = mode_count.get(mode, 0) + 1
        cand_count[t] = sum(1 for c in cands if c is not None)

        assignment = assign_tracks(cands, prev_tip, prev_center, prev_active)

        cur_tip = np.zeros((2, 2), dtype=np.float32)
        cur_center = np.zeros((2, 2), dtype=np.float32)
        cur_active = np.zeros(2, dtype=bool)

        for tr in range(2):
            ci = assignment[tr]
            if ci is None:
                continue
            cand = cands[ci]
            if cand is None:
                continue

            tips[tr, :, t] = cand["tip"]
            centers[tr, :, t] = cand["center"]
            presence[tr, t] = 1.0
            pca_ok[tr, t] = 1.0 if cand["pca_ok"] else 0.0
            area[tr, t] = cand["area"] / float(h * w)

            cur_tip[tr] = cand["tip"]
            cur_center[tr] = cand["center"]
            cur_active[tr] = True

        prev_tip = np.where(cur_active[:, None], cur_tip, prev_tip)
        prev_center = np.where(cur_active[:, None], cur_center, prev_center)
        prev_active = cur_active

        if (t + 1) % 2000 == 0:
            log(f"[case_{video_id}] {t+1}/{T}")

    geo = np.zeros((2, 7, T), dtype=np.float32)
    traj = np.zeros((2, 8, T), dtype=np.float32)

    ex = eye_center[:, 0]
    ey = eye_center[:, 1]
    r = np.maximum(eye_radius, 1e-6)

    for tr in range(2):
        px = tips[tr, 0]
        py = tips[tr, 1]
        pres = presence[tr] > 0.5

        dx = (px - ex) / r
        dy = (py - ey) / r
        dist = np.sqrt(dx * dx + dy * dy)

        geo[tr, 0] = np.where(pres, dx, 0.0)
        geo[tr, 1] = np.where(pres, dy, 0.0)
        geo[tr, 2] = np.where(pres, dist, 0.0)
        geo[tr, 3] = np.where(pres, px, 0.0)
        geo[tr, 4] = np.where(pres, py, 0.0)
        geo[tr, 5] = np.where(pres, area[tr] / np.maximum(eye_area, 1e-6), 0.0)
        geo[tr, 6] = pres.astype(np.float32)

        valid_transition = np.zeros(T, dtype=bool)
        valid_transition[1:] = pres[1:] & pres[:-1]

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

    # light-clip for safety, do not change presence/valid_transition
    geo[:, 0:2] = np.clip(np.nan_to_num(geo[:, 0:2], nan=0.0), -10, 10)
    geo[:, 2] = np.clip(np.nan_to_num(geo[:, 2], nan=0.0), 0, 10)
    traj[:, 0:2] = np.clip(np.nan_to_num(traj[:, 0:2], nan=0.0), -20, 20)
    traj[:, 2] = np.clip(np.nan_to_num(traj[:, 2], nan=0.0), 0, 20)
    traj[:, 3:6] = np.clip(np.nan_to_num(traj[:, 3:6], nan=0.0), -20, 20)
    traj[:, 6] = (traj[:, 6] > 0.5).astype(np.float32)
    traj[:, 7] = (traj[:, 7] > 0.5).astype(np.float32)

    out_npz = out_dir / f"case_{video_id}_v1c_cons_yolo_tiptrack_debug.npz"
    np.savez(
        out_npz,
        frame_ids=frame_ids,
        eye_center=eye_center.T.astype(np.float32),
        eye_radius=eye_radius.astype(np.float32),
        inst_tips=tips.astype(np.float32),
        inst_centers=centers.astype(np.float32),
        presence_mask=presence.astype(np.float32),
        geo=geo.astype(np.float32),
        traj=traj.astype(np.float32),
        pca_ok=pca_ok.astype(np.float32),
        cand_count=cand_count,
    )

    positive_dist = geo[:, 2][geo[:, 2] > 0]
    positive_speed = traj[:, 2][traj[:, 2] > 0]

    summary = {
        "video_id": video_id,
        "T": int(T),
        "conf_th": float(conf_th),
        "mode_count": mode_count,
        "cand_count_mean": float(cand_count.mean()),
        "cand_count_ge2_ratio": float((cand_count >= 2).mean()),
        "presence_mean": float(presence.mean()),
        "pca_ok_mean": float(pca_ok.mean()),
        "geo_dist_p95": float(np.percentile(positive_dist, 95)) if len(positive_dist) else 0.0,
        "geo_dist_max": float(np.max(positive_dist)) if len(positive_dist) else 0.0,
        "traj_speed_p95": float(np.percentile(positive_speed, 95)) if len(positive_speed) else 0.0,
        "traj_speed_max": float(np.max(positive_speed)) if len(positive_speed) else 0.0,
    }

    with open(out_dir / f"case_{video_id}_v1c_cons_yolo_tiptrack_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return out_npz


def read_frame(video_id, frame_id, size=(640, 480)):
    vp = CAT / "videos" / f"case_{video_id}.mp4"
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(vp)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_id}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame


def draw_points(rgb, z, t):
    out = rgb.copy()
    H, W = out.shape[:2]

    eye_center = z["eye_center"]
    eye_radius = z["eye_radius"]
    tips = z["inst_tips"]
    centers = z["inst_centers"]
    pres = z["presence_mask"]

    ex = int(round(eye_center[0, t] * W))
    ey = int(round(eye_center[1, t] * H))
    er = int(round(eye_radius[t] * min(W, H)))

    cv2.circle(out, (ex, ey), max(er, 2), (0, 255, 0), 2)
    cv2.circle(out, (ex, ey), 4, (0, 255, 0), -1)

    colors = [(255, 60, 20), (255, 190, 0)]

    for tr in range(2):
        if pres[tr, t] <= 0.5:
            continue

        tx = int(round(tips[tr, 0, t] * W))
        ty = int(round(tips[tr, 1, t] * H))
        cx = int(round(centers[tr, 0, t] * W))
        cy = int(round(centers[tr, 1, t] * H))

        color = colors[tr]
        cv2.circle(out, (cx, cy), 5, color, 1)
        cv2.circle(out, (tx, ty), 6, color, -1)
        cv2.line(out, (cx, cy), (tx, ty), color, 2)
        cv2.line(out, (ex, ey), (tx, ty), color, 1)

        t2 = min(t + 8, tips.shape[-1] - 1)
        if pres[tr, t2] > 0.5:
            fx = int(round(tips[tr, 0, t2] * W))
            fy = int(round(tips[tr, 1, t2] * H))
            cv2.arrowedLine(out, (tx, ty), (fx, fy), color, 2, tipLength=0.35)

        cv2.putText(out, f"track{tr}", (tx + 6, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return out


def visualize_frames(video_id, asset_path, out_dir, indices):
    z = np.load(asset_path)
    frame_ids = z["frame_ids"]

    for idx in indices:
        if idx < 0 or idx >= len(frame_ids):
            continue
        fid = int(frame_ids[idx])
        rgb = read_frame(video_id, fid)
        vis = draw_points(rgb, z, idx)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[0].axis("off")
        axes[1].imshow(vis)
        axes[1].set_title(f"v1c-cons YOLO-clean tip-track | local={idx}, frame={fid}")
        axes[1].axis("off")
        plt.tight_layout()

        out_path = out_dir / f"case_{video_id}_idx_{idx:06d}_frame_{fid:06d}_v1c_cons_yolo_tiptrack.png"
        plt.savefig(out_path, dpi=160)
        plt.close(fig)
        log(f"[OK] {out_path}")


def plot_curves(video_id, asset_path, out_dir, start=0, end=6000):
    z = np.load(asset_path)
    T = z["traj"].shape[-1]
    start = max(0, start)
    end = min(T, end)
    xs = np.arange(start, end)

    geo = z["geo"]
    traj = z["traj"]
    pres = z["presence_mask"]

    labels = []
    gtp = GT / f"case_{video_id}.txt"
    if gtp.exists():
        labels = [x.strip() for x in gtp.read_text(encoding="utf-8").splitlines() if x.strip()]
        labels = labels[:T]

    bounds = []
    for i in range(1, min(T, len(labels))):
        if labels[i] != labels[i - 1]:
            bounds.append(i)

    fig, axes = plt.subplots(7, 1, figsize=(18, 16), sharex=True)

    axes[0].set_title(f"case_{video_id} v1c-cons YOLO-clean tip-track distance / phase")
    for b in bounds:
        if start <= b < end:
            axes[0].axvline(b, color="k", alpha=0.15)
    for tr in range(2):
        y = np.where(pres[tr, start:end] > 0.5, geo[tr, 2, start:end], np.nan)
        axes[0].plot(xs, y, label=f"track{tr} dist/r")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    names = ["vx", "vy", "speed", "dist_delta", "dx_delta", "dy_delta"]
    for k in range(6):
        ax = axes[k + 1]
        ax.set_title(f"traj[{k}] {names[k]}")
        for tr in range(2):
            y = np.where(pres[tr, start:end] > 0.5, traj[tr, k, start:end], np.nan)
            ax.plot(xs, y, label=f"track{tr}", linewidth=1)
        for b in bounds:
            if start <= b < end:
                ax.axvline(b, color="k", alpha=0.15)
        ax.grid(alpha=0.2)
        ax.legend()

    axes[-1].set_xlabel("local frame index")
    plt.tight_layout()

    out_path = out_dir / f"case_{video_id}_v1c_cons_yolo_tiptrack_curves_{start}_{end}.png"
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    log(f"[OK] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", type=int, required=True)
    ap.add_argument("--indices", type=str, default="0,1200,2400,3600,4800")
    ap.add_argument("--curve_start", type=int, default=0)
    ap.add_argument("--curve_end", type=int, default=6000)
    ap.add_argument("--conf_th", type=float, default=0.30)
    ap.add_argument("--box_expand", type=float, default=0.08)
    ap.add_argument("--min_area", type=int, default=20)
    args = ap.parse_args()

    out_dir = OUT_ROOT / f"case_{args.video_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_path = build_tiptrack(
        args.video_id,
        out_dir,
        conf_th=args.conf_th,
        box_expand=args.box_expand,
        min_area=args.min_area
    )

    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    visualize_frames(args.video_id, asset_path, out_dir, indices)
    plot_curves(args.video_id, asset_path, out_dir, start=args.curve_start, end=args.curve_end)

    log(f"[DONE] outputs saved to {out_dir}")


if __name__ == "__main__":
    main()

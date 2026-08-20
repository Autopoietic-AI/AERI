import os
# -*- coding: utf-8 -*-
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))

DATA = ROOT / "data/sics155_100_geoig_r025"
YOLO = ROOT / "data/sics155_100_yolo_b2_multiinst_assets/npz"
SAM2 = ROOT / "data/sics155_100_sam2_masks_full/npz"

OUT_ROOT = ROOT / "data/sics155_100_a4_geo_assets_sam2_yolo_tiptrack_cons_compat"
OUT_NPZ = OUT_ROOT / "npz"
OUT_META = OUT_ROOT / "meta"

MAX_INST = 2
CLS_PROXY_DIM = 512
EPS = 1e-6


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
    p = DATA / "groundTruth" / f"{video}.txt"
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def safe_load(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def xyxy_center(boxes):
    """
    boxes: (T, K, 4), absolute pixel xyxy.
    return centers: (T, K, 2)
    """
    c = np.zeros(boxes.shape[:2] + (2,), dtype=np.float32)
    c[..., 0] = (boxes[..., 0] + boxes[..., 2]) * 0.5
    c[..., 1] = (boxes[..., 1] + boxes[..., 3]) * 0.5
    return c.astype(np.float32)


def estimate_eye_center_radius(sam, yolo):
    """
    output:
      eye_center: (2,T), normalized 0-1
      eye_radius: (T,), normalized by max(W,H)
    """
    frame_ids = sam["frame_ids"]
    T = len(frame_ids)

    width = float(sam["width"]) if "width" in sam else float(yolo["width"])
    height = float(sam["height"]) if "height" in sam else float(yolo["height"])
    scale = max(width, height)

    eye_valid = sam["eye_valid"].astype(np.float32) if "eye_valid" in sam else yolo["eye_valid"].astype(np.float32)

    # prefer SAM mask center if available; it is already normalized in our SICS SAM2 script.
    if "eye_center" in sam:
        ec = sam["eye_center"].astype(np.float32)  # (T,2), normalized
    else:
        box = yolo["eye_xyxy"].astype(np.float32)
        ec = np.zeros((T, 2), dtype=np.float32)
        ec[:, 0] = ((box[:, 0] + box[:, 2]) * 0.5) / max(width - 1.0, 1.0)
        ec[:, 1] = ((box[:, 1] + box[:, 3]) * 0.5) / max(height - 1.0, 1.0)

    if "eye_xyxy" in sam:
        eye_box = sam["eye_xyxy"].astype(np.float32)
    else:
        eye_box = yolo["eye_xyxy"].astype(np.float32)

    bw = np.maximum(0.0, eye_box[:, 2] - eye_box[:, 0])
    bh = np.maximum(0.0, eye_box[:, 3] - eye_box[:, 1])
    r = 0.5 * np.sqrt(bw * bh) / max(scale, 1.0)

    # if invalid, keep center/radius zero
    ec[eye_valid <= 0] = 0
    r[eye_valid <= 0] = 0

    return ec.T.astype(np.float32), r.astype(np.float32)


def estimate_inst_centers_tips(sam, yolo):
    """
    output:
      inst_centers: (2,2,T), normalized 0-1
      inst_tips:    (2,2,T), normalized 0-1

    For SICS first version:
      - centers from SAM mask center if union center available is only one; for multi-inst use YOLO box centers.
      - tips approximated by the point on the bbox closer to eye center.
        This is a conservative proxy when true tip PCA is not available.
    """
    T = len(sam["frame_ids"])

    width = float(sam["width"]) if "width" in sam else float(yolo["width"])
    height = float(sam["height"]) if "height" in sam else float(yolo["height"])

    if "inst_xyxy_multi" in yolo:
        boxes = yolo["inst_xyxy_multi"].astype(np.float32)
        valid = yolo["inst_valid_multi"].astype(np.float32)
    elif "inst_xyxy_multi" in sam:
        boxes = sam["inst_xyxy_multi"].astype(np.float32)
        valid = sam["inst_valid_multi"].astype(np.float32)
    else:
        boxes = yolo["inst_xyxy"][:, None, :].astype(np.float32)
        valid = yolo["inst_valid"][:, None].astype(np.float32)

    K = min(MAX_INST, boxes.shape[1])
    boxes = boxes[:, :K, :]
    valid = valid[:, :K]

    if K < MAX_INST:
        pad_boxes = np.zeros((T, MAX_INST - K, 4), dtype=np.float32)
        pad_valid = np.zeros((T, MAX_INST - K), dtype=np.float32)
        boxes = np.concatenate([boxes, pad_boxes], axis=1)
        valid = np.concatenate([valid, pad_valid], axis=1)

    centers_abs = xyxy_center(boxes)  # (T,2,2)

    centers = np.zeros((MAX_INST, 2, T), dtype=np.float32)
    tips = np.zeros((MAX_INST, 2, T), dtype=np.float32)

    # eye center for selecting nearest corner as tip proxy
    eye_center, _ = estimate_eye_center_radius(sam, yolo)
    eye_xy = np.zeros((T, 2), dtype=np.float32)
    eye_xy[:, 0] = eye_center[0] * max(width - 1.0, 1.0)
    eye_xy[:, 1] = eye_center[1] * max(height - 1.0, 1.0)

    for k in range(MAX_INST):
        # center normalized
        centers[k, 0, :] = centers_abs[:, k, 0] / max(width - 1.0, 1.0)
        centers[k, 1, :] = centers_abs[:, k, 1] / max(height - 1.0, 1.0)

        # four bbox corners; choose closest to eye center as tip proxy
        x1, y1, x2, y2 = boxes[:, k, 0], boxes[:, k, 1], boxes[:, k, 2], boxes[:, k, 3]
        corners = np.stack([
            np.stack([x1, y1], axis=1),
            np.stack([x1, y2], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
        ], axis=1)  # (T,4,2)

        d = ((corners - eye_xy[:, None, :]) ** 2).sum(axis=2)
        j = np.argmin(d, axis=1)
        chosen = corners[np.arange(T), j]

        tips[k, 0, :] = chosen[:, 0] / max(width - 1.0, 1.0)
        tips[k, 1, :] = chosen[:, 1] / max(height - 1.0, 1.0)

        invalid = valid[:, k] <= 0
        centers[k, :, invalid] = 0
        tips[k, :, invalid] = 0

    return centers.astype(np.float32), tips.astype(np.float32), valid.astype(np.float32)


def build_geo_traj_aux(video, sam, yolo):
    T = len(sam["frame_ids"])
    T_gt = gt_len(video)
    if T != T_gt:
        raise RuntimeError(f"{video}: SAM2 T={T}, GT T={T_gt}")

    frame_ids = sam["frame_ids"].astype(np.int32)

    eye_valid = sam["eye_valid"].astype(np.float32)
    inst_valid_multi = sam["inst_valid_multi"].astype(np.float32) if "inst_valid_multi" in sam else yolo["inst_valid_multi"].astype(np.float32)
    inst_valid_multi = inst_valid_multi[:, :MAX_INST]

    if inst_valid_multi.shape[1] < MAX_INST:
        pad = np.zeros((T, MAX_INST - inst_valid_multi.shape[1]), dtype=np.float32)
        inst_valid_multi = np.concatenate([inst_valid_multi, pad], axis=1)

    inst_valid_any = (inst_valid_multi.max(axis=1) > 0).astype(np.float32)

    eye_area = sam["eye_area"].astype(np.float32) if "eye_area" in sam else np.zeros((T,), dtype=np.float32)
    inst_area_union = sam["inst_area"].astype(np.float32) if "inst_area" in sam else np.zeros((T,), dtype=np.float32)

    # no per-instrument SAM area in our first packed masks, so distribute union area to valid instruments
    inst_area_multi = np.zeros((T, MAX_INST), dtype=np.float32)
    denom = np.maximum(inst_valid_multi.sum(axis=1, keepdims=True), 1.0)
    inst_area_multi = inst_valid_multi * (inst_area_union[:, None] / denom)

    eye_center, eye_radius = estimate_eye_center_radius(sam, yolo)
    inst_centers, inst_tips, inst_valid_multi = estimate_inst_centers_tips(sam, yolo)

    presence_mask = inst_valid_multi.T.astype(np.float32)  # (2,T)

    geo = np.zeros((MAX_INST, 7, T), dtype=np.float32)

    # normalized relation features
    # use radius in normalized max-dimension space. Since centers/tips are normalized separately by W/H,
    # approximate radius in normalized image coordinate; avoid zero.
    er = np.maximum(eye_radius, EPS)

    ex = eye_center[0]
    ey = eye_center[1]

    for k in range(MAX_INST):
        px = inst_tips[k, 0]
        py = inst_tips[k, 1]
        cx = inst_centers[k, 0]
        cy = inst_centers[k, 1]

        valid = presence_mask[k] * eye_valid

        dx = (px - ex) / er
        dy = (py - ey) / er
        dist = np.sqrt(dx * dx + dy * dy)
        angle = np.arctan2(dy, dx) / np.pi

        cdx = (cx - ex) / er
        cdy = (cy - ey) / er
        cdist = np.sqrt(cdx * cdx + cdy * cdy)

        # inside / near operation region proxy
        near_eye = (dist < 1.5).astype(np.float32)

        geo[k, 0, :] = dx
        geo[k, 1, :] = dy
        geo[k, 2, :] = dist
        geo[k, 3, :] = angle
        geo[k, 4, :] = cdx
        geo[k, 5, :] = cdy
        geo[k, 6, :] = near_eye

        geo[k, :, valid <= 0] = 0

    traj = np.zeros((MAX_INST, 8, T), dtype=np.float32)

    for k in range(MAX_INST):
        px = inst_tips[k, 0]
        py = inst_tips[k, 1]
        valid = presence_mask[k] * eye_valid

        vx = np.zeros((T,), dtype=np.float32)
        vy = np.zeros((T,), dtype=np.float32)
        vx[1:] = px[1:] - px[:-1]
        vy[1:] = py[1:] - py[:-1]

        speed = np.sqrt(vx * vx + vy * vy)
        acc = np.zeros((T,), dtype=np.float32)
        acc[1:] = speed[1:] - speed[:-1]

        dx = geo[k, 0]
        dy = geo[k, 1]
        dist = geo[k, 2]
        ddist = np.zeros((T,), dtype=np.float32)
        ddist[1:] = dist[1:] - dist[:-1]

        # validity transition cues
        vprev = np.zeros((T,), dtype=np.float32)
        vprev[1:] = valid[:-1]
        appear = ((valid > 0) & (vprev <= 0)).astype(np.float32)
        disappear = ((valid <= 0) & (vprev > 0)).astype(np.float32)

        traj[k, 0, :] = vx
        traj[k, 1, :] = vy
        traj[k, 2, :] = speed
        traj[k, 3, :] = acc
        traj[k, 4, :] = ddist
        traj[k, 5, :] = valid
        traj[k, 6, :] = appear
        traj[k, 7, :] = disappear

        traj[k, :, valid <= 0] = 0

    # aux: compact reliability / mask cues
    eye_conf = yolo["eye_conf"].astype(np.float32) if "eye_conf" in yolo else eye_valid
    if "inst_conf_multi" in yolo:
        inst_conf_multi = yolo["inst_conf_multi"].astype(np.float32)[:, :MAX_INST]
        inst_conf = inst_conf_multi.max(axis=1)
    elif "inst_conf" in yolo:
        inst_conf = yolo["inst_conf"].astype(np.float32)
    else:
        inst_conf = inst_valid_any

    aux = np.zeros((4, T), dtype=np.float32)
    aux[0] = eye_valid * eye_conf
    aux[1] = inst_valid_any * inst_conf
    aux[2] = eye_area
    aux[3] = inst_area_union

    # cls_proxy is kept as compatibility field.
    # It is a weak placeholder in SICS first version; downstream A5 should mainly use geo/traj/aux/presence.
    cls_proxy = np.zeros((MAX_INST, CLS_PROXY_DIM, T), dtype=np.float32)
    cls_proxy_mask = presence_mask.copy().astype(np.float32)

    return {
        "geo": geo.astype(np.float32),
        "aux": aux.astype(np.float32),
        "presence_mask": presence_mask.astype(np.float32),
        "cls_proxy": cls_proxy.astype(np.float32),
        "cls_proxy_mask": cls_proxy_mask.astype(np.float32),
        "traj": traj.astype(np.float32),
        "eye_center": eye_center.astype(np.float32),
        "eye_radius": eye_radius.astype(np.float32),
        "inst_tips": inst_tips.astype(np.float32),
        "frame_ids": frame_ids.astype(np.int32),
        "eye_valid": eye_valid.astype(np.float32),
        "inst_valid_multi": inst_valid_multi.astype(np.float32),
        "inst_valid_any": inst_valid_any.astype(np.float32),
        "eye_area": eye_area.astype(np.float32),
        "inst_area_multi": inst_area_multi.astype(np.float32),
        "inst_area_union": inst_area_union.astype(np.float32),
        "inst_centers": inst_centers.astype(np.float32),
    }


def process_video(video, overwrite=False):
    out_npz = OUT_NPZ / f"{video}.npz"
    out_meta = OUT_META / f"{video}_summary.json"

    if out_npz.exists() and out_meta.exists() and not overwrite:
        z = np.load(out_npz)
        T = gt_len(video)
        if "geo" in z and z["geo"].shape[-1] == T:
            print(f"[SKIP] {video}: exists", flush=True)
            return {"video": video, "status": "skip", "npz": str(out_npz)}

    sam = safe_load(SAM2 / f"{video}.npz")
    yolo = safe_load(YOLO / f"{video}.npz")

    assets = build_geo_traj_aux(video, sam, yolo)

    np.savez_compressed(out_npz, **assets)

    T = assets["frame_ids"].shape[0]
    meta = {
        "video": video,
        "status": "ok",
        "T": int(T),
        "geo_shape": list(assets["geo"].shape),
        "aux_shape": list(assets["aux"].shape),
        "traj_shape": list(assets["traj"].shape),
        "presence_shape": list(assets["presence_mask"].shape),
        "eye_valid_ratio": float(assets["eye_valid"].mean()),
        "inst_valid_any_ratio": float(assets["inst_valid_any"].mean()),
        "eye_radius_mean_valid": float(assets["eye_radius"][assets["eye_valid"] > 0].mean()) if np.any(assets["eye_valid"] > 0) else 0.0,
        "npz": str(out_npz),
    }

    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[OK] {video}: T={T}, geo={assets['geo'].shape}, "
        f"eye={meta['eye_valid_ratio']:.4f}, inst={meta['inst_valid_any_ratio']:.4f}",
        flush=True,
    )

    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="all")
    ap.add_argument("--video", default=None)
    ap.add_argument("--limit", type=int, default=0)
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
    print("[SICS155_100 A4 GeoIG assets: YOLO tiptrack conservative compat]", flush=True)
    print("DATA:", DATA, flush=True)
    print("YOLO:", YOLO, flush=True)
    print("SAM2:", SAM2, flush=True)
    print("OUT:", OUT_ROOT, flush=True)
    print("videos:", len(videos), videos[:10], "...", flush=True)
    print("=" * 80, flush=True)

    summary_path = OUT_META / "summary_all.jsonl"
    if args.overwrite and summary_path.exists():
        summary_path.unlink()

    for i, video in enumerate(videos, start=1):
        print(f"\n===== [{i}/{len(videos)}] {video} =====", flush=True)
        try:
            res = process_video(video, overwrite=args.overwrite)
        except Exception as e:
            res = {
                "video": video,
                "status": "error",
                "error": repr(e),
            }
            print("[ERROR]", json.dumps(res, ensure_ascii=False), flush=True)

        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("[DONE]", flush=True)
    print("Output:", OUT_NPZ, flush=True)
    print("Summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()

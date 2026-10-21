# -*- coding: utf-8 -*-
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


ROOT = Path("/root/autodl-tmp/MS-TCN2")

DATA = ROOT / "data/sics155_100_geoig_r025"
MANIFEST = DATA / "frame_manifest"
YOLO = ROOT / "data/sics155_100_yolo_b2_multiinst_assets"
OUT = ROOT / "data/sics155_100_sam2_masks_full"

SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT = "/root/autodl-tmp/sam2_repo/checkpoints/sam2.1_hiera_small.pt"

MASK_H = 224
MASK_W = 224
PACK_LEN = (MASK_H * MASK_W + 7) // 8


def prepare_out():
    for sub in ["npz", "vis", "meta", "groundTruth", "splits"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    shutil.copy2(DATA / "mapping.txt", OUT / "mapping.txt")

    for p in (DATA / "groundTruth").glob("*.txt"):
        q = OUT / "groundTruth" / p.name
        if not q.exists():
            shutil.copy2(p, q)

    for p in (DATA / "splits").glob("*.bundle"):
        q = OUT / "splits" / p.name
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


def read_manifest(video):
    p = MANIFEST / f"{video}.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def clamp_box(box, W, H):
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(W - 1.0, x1))
    x2 = max(0.0, min(W - 1.0, x2))
    y1 = max(0.0, min(H - 1.0, y1))
    y2 = max(0.0, min(H - 1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def expand_box(box, W, H, ratio=0.08):
    box = clamp_box(box, W, H)
    if box is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * ratio
    x2 += bw * ratio
    y1 -= bh * ratio
    y2 += bh * ratio
    return clamp_box([x1, y1, x2, y2], W, H)


def rect_mask_for_box(box, H, W):
    m = np.zeros((H, W), dtype=bool)
    box = clamp_box(box, W, H)
    if box is None:
        return m
    x1, y1, x2, y2 = box.astype(int).tolist()
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 > x1 and y2 > y1:
        m[y1:y2 + 1, x1:x2 + 1] = True
    return m


def run_sam_one_box(predictor, box, H, W, box_expand=0.08, clip_to_box=True):
    box = clamp_box(box, W, H)
    if box is None:
        return np.zeros((H, W), dtype=np.uint8), 0.0

    try:
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=False,
        )
    except Exception:
        masks, scores, _ = predictor.predict(
            box=box[None, :],
            multimask_output=False,
        )

    if masks is None or len(masks) == 0:
        return np.zeros((H, W), dtype=np.uint8), 0.0

    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)

    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]

    best = int(np.argmax(scores)) if len(scores) else 0
    mask = masks[best].astype(bool)
    score = float(scores[best]) if len(scores) else 0.0

    if clip_to_box:
        clip_box = expand_box(box, W, H, ratio=box_expand)
        if clip_box is not None:
            mask = mask & rect_mask_for_box(clip_box, H, W)

    return mask.astype(np.uint8), score


def resize_and_pack(mask):
    m = cv2.resize(mask.astype(np.uint8), (MASK_W, MASK_H), interpolation=cv2.INTER_NEAREST)
    flat = m.reshape(-1).astype(np.uint8)
    return np.packbits(flat, bitorder="little")


def mask_stats(mask):
    if mask is None or mask.sum() == 0:
        return 0.0, np.array([0.0, 0.0], dtype=np.float32)

    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    area = float(mask.mean())
    cx = float(xs.mean() / max(W - 1, 1))
    cy = float(ys.mean() / max(H - 1, 1))
    return area, np.array([cx, cy], dtype=np.float32)


def overlay_masks(frame, eye_mask, inst_mask):
    overlay = frame.copy()
    overlay[eye_mask > 0] = (0, 255, 0)
    overlay[inst_mask > 0] = (0, 0, 255)
    vis = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

    for m, color in [(eye_mask, (0, 255, 0)), (inst_mask, (0, 0, 255))]:
        cnts, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)

    return vis


def process_video(video, predictor, args):
    out_npz = OUT / "npz" / f"{video}.npz"
    out_meta = OUT / "meta" / f"{video}_sam2_full.json"
    vis_dir = OUT / "vis" / video

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = read_manifest(video)
    T = len(frame_paths)
    T_gt = gt_len(video)

    if T != T_gt:
        raise RuntimeError(f"{video}: manifest T={T}, gt T={T_gt}")

    if out_npz.exists() and not args.overwrite:
        z_old = np.load(out_npz)
        if "frame_ids" in z_old and len(z_old["frame_ids"]) == T and (
            "eye_masks_packed" in z_old or "eye_packed" in z_old
        ):
            print(f"[SKIP] {video}: exists, T={T}", flush=True)
            return {"video": video, "status": "skip", "T": int(T), "npz": str(out_npz)}

    yolo_path = YOLO / "npz" / f"{video}.npz"
    if not yolo_path.exists():
        raise FileNotFoundError(yolo_path)

    y = np.load(yolo_path, allow_pickle=True)

    yolo_frame_ids = y["frame_ids"]
    if len(yolo_frame_ids) != T:
        raise RuntimeError(f"{video}: yolo T={len(yolo_frame_ids)}, expected T={T}")

    eye_xyxy = y["eye_xyxy"]
    eye_valid_yolo = y["eye_valid"]
    eye_conf = y["eye_conf"]

    if "inst_xyxy_multi" in y and "inst_valid_multi" in y:
        inst_xyxy_multi = y["inst_xyxy_multi"]
        inst_valid_multi = y["inst_valid_multi"]
        inst_conf_multi = y["inst_conf_multi"]
        max_inst = inst_xyxy_multi.shape[1]
    else:
        inst_xyxy_multi = y["inst_xyxy"][:, None, :]
        inst_valid_multi = y["inst_valid"][:, None]
        inst_conf_multi = y["inst_conf"][:, None]
        max_inst = 1

    frame_ids = np.arange(T, dtype=np.int32)

    eye_packed = np.zeros((T, PACK_LEN), dtype=np.uint8)
    inst_packed = np.zeros((T, PACK_LEN), dtype=np.uint8)

    eye_valid = np.zeros((T,), dtype=np.float32)
    inst_valid = np.zeros((T,), dtype=np.float32)

    eye_sam_score = np.zeros((T,), dtype=np.float32)
    inst_sam_score = np.zeros((T,), dtype=np.float32)

    eye_area = np.zeros((T,), dtype=np.float32)
    inst_area = np.zeros((T,), dtype=np.float32)

    eye_center = np.zeros((T, 2), dtype=np.float32)
    inst_center = np.zeros((T, 2), dtype=np.float32)

    width = 0
    height = 0
    saved_vis = 0

    for t, fp in enumerate(frame_paths):
        frame = cv2.imread(fp)
        if frame is None:
            raise RuntimeError(f"{video}: cannot read frame t={t}: {fp}")

        H, W = frame.shape[:2]
        if width == 0:
            width = W
            height = H

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

        # eye mask
        eye_mask = np.zeros((H, W), dtype=np.uint8)
        if eye_valid_yolo[t] > 0:
            eye_mask, eye_score = run_sam_one_box(
                predictor,
                eye_xyxy[t],
                H,
                W,
                box_expand=args.box_expand,
                clip_to_box=bool(args.clip_to_box),
            )
            eye_sam_score[t] = eye_score
            if eye_mask.sum() > 0:
                eye_valid[t] = 1.0

        # instrument union mask
        inst_union = np.zeros((H, W), dtype=np.uint8)
        inst_scores = []

        for k in range(max_inst):
            if inst_valid_multi[t, k] <= 0:
                continue

            m, score = run_sam_one_box(
                predictor,
                inst_xyxy_multi[t, k],
                H,
                W,
                box_expand=args.box_expand,
                clip_to_box=bool(args.clip_to_box),
            )
            if m.sum() > 0:
                inst_union = np.maximum(inst_union, m)
                inst_scores.append(score)

        if inst_union.sum() > 0:
            inst_valid[t] = 1.0
            inst_sam_score[t] = float(np.mean(inst_scores)) if inst_scores else 0.0

        eye_packed[t] = resize_and_pack(eye_mask)
        inst_packed[t] = resize_and_pack(inst_union)

        ea, ec = mask_stats(eye_mask)
        ia, ic = mask_stats(inst_union)
        eye_area[t] = ea
        inst_area[t] = ia
        eye_center[t] = ec
        inst_center[t] = ic

        if args.save_vis and (t % args.vis_stride == 0):
            vis = overlay_masks(frame, eye_mask, inst_union)
            phase = Path(fp).parent.name
            save_path = vis_dir / f"t{t:06d}_{phase}_{Path(fp).stem}.jpg"
            cv2.imwrite(str(save_path), vis)
            saved_vis += 1

        if args.progress_every > 0 and (t + 1) % args.progress_every == 0:
            print(
                f"[PROGRESS] {video}: {t+1}/{T} "
                f"eye={eye_valid[:t+1].mean():.4f} inst={inst_valid[:t+1].mean():.4f}",
                flush=True,
            )

    np.savez_compressed(
        out_npz,
        frame_ids=frame_ids,
        frame_paths=np.asarray(frame_paths),
        width=np.array(width, dtype=np.int32),
        height=np.array(height, dtype=np.int32),
        mask_h=np.array(MASK_H, dtype=np.int32),
        mask_w=np.array(MASK_W, dtype=np.int32),
        pack_len=np.array(PACK_LEN, dtype=np.int32),

        # compatibility names
        eye_packed=eye_packed,
        inst_packed=inst_packed,
        eye_masks_packed=eye_packed,
        inst_masks_packed=inst_packed,

        eye_valid=eye_valid,
        inst_valid=inst_valid,
        eye_sam_score=eye_sam_score,
        inst_sam_score=inst_sam_score,
        eye_area=eye_area,
        inst_area=inst_area,
        eye_center=eye_center,
        inst_center=inst_center,

        # carry YOLO information forward
        eye_xyxy=eye_xyxy,
        eye_conf=eye_conf,
        eye_valid_yolo=eye_valid_yolo,
        inst_xyxy_multi=inst_xyxy_multi,
        inst_valid_multi=inst_valid_multi,
        inst_conf_multi=inst_conf_multi,
    )

    meta = {
        "video": video,
        "T": int(T),
        "width": int(width),
        "height": int(height),
        "mask_h": MASK_H,
        "mask_w": MASK_W,
        "eye_valid_ratio": float(eye_valid.mean()),
        "inst_valid_ratio": float(inst_valid.mean()),
        "eye_yolo_valid_ratio": float(eye_valid_yolo.mean()),
        "inst_yolo_valid_ratio": float(inst_valid_multi.max(axis=1).mean()),
        "eye_sam_score_mean_valid": float(eye_sam_score[eye_valid > 0].mean()) if np.any(eye_valid > 0) else 0.0,
        "inst_sam_score_mean_valid": float(inst_sam_score[inst_valid > 0].mean()) if np.any(inst_valid > 0) else 0.0,
        "eye_area_mean_valid": float(eye_area[eye_valid > 0].mean()) if np.any(eye_valid > 0) else 0.0,
        "inst_area_mean_valid": float(inst_area[inst_valid > 0].mean()) if np.any(inst_valid > 0) else 0.0,
        "saved_vis": int(saved_vis),
        "npz": str(out_npz),
    }

    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[OK] {video}: T={T}, eye_valid={meta['eye_valid_ratio']:.4f}, "
        f"inst_valid={meta['inst_valid_ratio']:.4f}, saved_vis={saved_vis}",
        flush=True,
    )

    return {"video": video, "status": "ok", **meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="all", help="all or split bundle name")
    ap.add_argument("--video", default=None, help="single video id, e.g. BL_0101")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--box_expand", type=float, default=0.08)
    ap.add_argument("--clip_to_box", type=int, default=1)
    ap.add_argument("--save_vis", type=int, default=0)
    ap.add_argument("--vis_stride", type=int, default=3000)
    ap.add_argument("--progress_every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
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
    print("[SICS155_100 SAM2 FULL PACKED MASK BUILDER]", flush=True)
    print("DATA:", DATA, flush=True)
    print("YOLO:", YOLO, flush=True)
    print("OUT:", OUT, flush=True)
    print("SAM2_CFG:", SAM2_CFG, flush=True)
    print("SAM2_CKPT:", SAM2_CKPT, flush=True)
    print("videos:", len(videos), videos[:10], "...", flush=True)
    print("box_expand:", args.box_expand, "clip_to_box:", args.clip_to_box, flush=True)
    print("=" * 80, flush=True)

    sam2_model = build_sam2(
        config_file=SAM2_CFG,
        ckpt_path=SAM2_CKPT,
        device=args.device,
    )

    predictor = SAM2ImagePredictor(sam2_model)

    summary_path = OUT / "meta" / "summary_all.jsonl"
    if args.overwrite and summary_path.exists():
        summary_path.unlink()

    for i, video in enumerate(videos, start=1):
        print(f"\n===== [{i}/{len(videos)}] {video} =====", flush=True)
        try:
            res = process_video(video, predictor, args)
        except Exception as e:
            res = {
                "video": video,
                "status": "error",
                "error": repr(e),
            }
            print("[ERROR]", json.dumps(res, ensure_ascii=False), flush=True)

        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

        torch.cuda.empty_cache()

    print("[DONE]", flush=True)
    print("Output:", OUT, flush=True)
    print("Summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()

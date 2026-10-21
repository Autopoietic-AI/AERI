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
CAT = Path("/root/autodl-tmp/101data/cataract-101")

A0 = ROOT / "data/cataract101_m10_a0_rgb2048"
YOLO = ROOT / "data/cataract101_m10_yolo_b2_multiinst_assets"
OUT = ROOT / "data/cataract101_m10_a2c_sam2_masks_full"

SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT = "/root/autodl-tmp/sam2_repo/checkpoints/sam2.1_hiera_small.pt"

MASK_H = 224
MASK_W = 224
PACK_LEN = (MASK_H * MASK_W + 7) // 8


def prepare_out():
    for sub in ["npz", "vis", "meta", "groundTruth", "splits"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    shutil.copy2(A0 / "mapping.txt", OUT / "mapping.txt")

    for p in (A0 / "groundTruth").glob("case_*.txt"):
        q = OUT / "groundTruth" / p.name
        if not q.exists():
            shutil.copy2(p, q)

    for p in (A0 / "splits").glob("*.bundle"):
        q = OUT / "splits" / p.name
        if not q.exists():
            shutil.copy2(p, q)


def read_ids(bundle):
    ids = []
    with open(A0 / "splits" / bundle, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("case_") and s.endswith(".txt"):
                ids.append(int(s.replace("case_", "").replace(".txt", "")))
    return ids


def gt_len(vid):
    p = A0 / "groundTruth" / f"case_{vid}.txt"
    return sum(1 for _ in open(p, encoding="utf-8"))


def clamp_box(box, W, H):
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


def process_video(vid, predictor, args):
    out_npz = OUT / "npz" / f"case_{vid}.npz"
    out_meta = OUT / "meta" / f"case_{vid}_sam2_full.json"
    vis_dir = OUT / "vis" / f"case_{vid}"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    T = gt_len(vid)

    if out_npz.exists() and not args.overwrite:
        z_old = np.load(out_npz)
        if "frame_ids" in z_old and len(z_old["frame_ids"]) == T:
            print(f"[SKIP] case_{vid}: exists, T={T}")
            return

    yolo_path = YOLO / "npz" / f"case_{vid}.npz"
    if not yolo_path.exists():
        raise FileNotFoundError(yolo_path)

    z = np.load(yolo_path)
    frame_ids = z["frame_ids"]

    if len(frame_ids) != T:
        raise RuntimeError(f"case_{vid}: YOLO T={len(frame_ids)} != GT T={T}")

    start = int(frame_ids[0])
    video_path = CAT / "videos" / f"case_{vid}.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    eye_packed = []
    inst_union_packed = []

    eye_score = np.zeros((T,), dtype=np.float32)
    inst_score_multi = np.zeros((T, 2), dtype=np.float32)

    eye_area = np.zeros((T,), dtype=np.float32)
    inst_area_union = np.zeros((T,), dtype=np.float32)
    inst_area_multi = np.zeros((T, 2), dtype=np.float32)

    eye_center = np.zeros((T, 2), dtype=np.float32)
    inst_center_multi = np.zeros((T, 2, 2), dtype=np.float32)

    eye_valid = z["eye_valid"].astype(np.float32)
    inst_valid_multi = z["inst_valid_multi"].astype(np.float32)

    idx = 0
    used = 0
    saved_vis = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx >= start and used < T:
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictor.set_image(rgb)

            # eye
            if eye_valid[used] > 0:
                emask, escore = run_sam_one_box(
                    predictor,
                    z["eye_xyxy"][used],
                    H,
                    W,
                    box_expand=args.box_expand,
                    clip_to_box=bool(args.clip_to_box),
                )
            else:
                emask = np.zeros((H, W), dtype=np.uint8)
                escore = 0.0

            eye_score[used] = escore
            eye_area[used], eye_center[used] = mask_stats(emask)

            # instruments top-2, union for X5
            inst_union = np.zeros((H, W), dtype=np.uint8)
            boxes = z["inst_xyxy_multi"][used]
            valids = z["inst_valid_multi"][used]

            for k in range(min(2, boxes.shape[0])):
                if valids[k] <= 0:
                    continue

                imask, iscore = run_sam_one_box(
                    predictor,
                    boxes[k],
                    H,
                    W,
                    box_expand=args.box_expand,
                    clip_to_box=bool(args.clip_to_box),
                )
                inst_union = np.maximum(inst_union, imask)

                inst_score_multi[used, k] = iscore
                inst_area_multi[used, k], inst_center_multi[used, k] = mask_stats(imask)

            inst_area_union[used], _ = mask_stats(inst_union)

            eye_packed.append(resize_and_pack(emask))
            inst_union_packed.append(resize_and_pack(inst_union))

            if args.save_vis and used % args.vis_stride == 0:
                vis = overlay_masks(frame, emask, inst_union)
                title = f"case_{vid} frame={idx} eye_area={eye_area[used]:.3f} inst_area={inst_area_union[used]:.3f}"
                cv2.putText(vis, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 3)
                cv2.putText(vis, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0,0,0), 1)
                cv2.imwrite(str(vis_dir / f"frame_{idx:06d}_overlay.jpg"), vis)
                saved_vis += 1

            used += 1

            if used % args.progress_every == 0:
                print(
                    f"[PROGRESS] case_{vid}: {used}/{T}, "
                    f"eye_area={eye_area[max(0, used-args.progress_every):used].mean():.4f}, "
                    f"inst_area={inst_area_union[max(0, used-args.progress_every):used].mean():.4f}"
                )

        idx += 1
        if used >= T:
            break

    cap.release()

    if used != T:
        raise RuntimeError(f"case_{vid}: used={used}, expected T={T}")

    eye_packed = np.stack(eye_packed).astype(np.uint8)
    inst_union_packed = np.stack(inst_union_packed).astype(np.uint8)

    np.savez_compressed(
        out_npz,
        frame_ids=frame_ids.astype(np.int32),
        start_frame=np.array(start, dtype=np.int32),
        mask_h=np.array(MASK_H, dtype=np.int32),
        mask_w=np.array(MASK_W, dtype=np.int32),
        pack_len=np.array(PACK_LEN, dtype=np.int32),

        eye_mask_packed=eye_packed,
        inst_union_mask_packed=inst_union_packed,

        eye_valid=eye_valid,
        inst_valid_multi=inst_valid_multi,
        inst_valid_any=(inst_valid_multi.max(axis=1) > 0).astype(np.float32),

        eye_score=eye_score,
        inst_score_multi=inst_score_multi,

        eye_area=eye_area,
        inst_area_union=inst_area_union,
        inst_area_multi=inst_area_multi,

        eye_center=eye_center,
        inst_center_multi=inst_center_multi,
    )

    meta = {
        "video_id": int(vid),
        "T": int(T),
        "start_frame": int(start),
        "mask_size": [MASK_H, MASK_W],
        "sam2_cfg": SAM2_CFG,
        "sam2_ckpt": SAM2_CKPT,
        "yolo_path": str(yolo_path),
        "out_npz": str(out_npz),
        "mean_eye_area": float(eye_area.mean()),
        "mean_inst_area_union": float(inst_area_union.mean()),
        "mean_eye_score": float(eye_score[eye_valid > 0].mean()) if eye_valid.sum() > 0 else 0.0,
        "mean_inst_score": float(inst_score_multi[inst_valid_multi > 0].mean()) if inst_valid_multi.sum() > 0 else 0.0,
        "saved_vis": int(saved_vis),
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] case_{vid}: T={T}, "
        f"eye_area={meta['mean_eye_area']:.4f}, "
        f"inst_area={meta['mean_inst_area_union']:.4f}, "
        f"vis={saved_vis}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", type=int, default=None)
    ap.add_argument("--bundle", default="all.bundle")
    ap.add_argument("--box_expand", type=float, default=0.08)
    ap.add_argument("--clip_to_box", type=int, default=1)
    ap.add_argument("--save_vis", type=int, default=1)
    ap.add_argument("--vis_stride", type=int, default=3000)
    ap.add_argument("--progress_every", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prepare_out()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("SAM2_CFG:", SAM2_CFG)
    print("SAM2_CKPT:", SAM2_CKPT)
    print("OUT:", OUT)

    sam2_model = build_sam2(
        SAM2_CFG,
        SAM2_CKPT,
        device=device,
        apply_postprocessing=False,
    )
    predictor = SAM2ImagePredictor(sam2_model)

    vids = [args.video_id] if args.video_id is not None else read_ids(args.bundle)
    print("num videos:", len(vids))

    for i, vid in enumerate(vids, 1):
        print(f"\n===== [{i}/{len(vids)}] case_{vid} =====")
        process_video(vid, predictor, args)


if __name__ == "__main__":
    main()

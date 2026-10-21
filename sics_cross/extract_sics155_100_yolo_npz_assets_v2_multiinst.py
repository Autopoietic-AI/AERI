# -*- coding: utf-8 -*-
import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path("/root/autodl-tmp/MS-TCN2")

DATA = ROOT / "data/sics155_100_geoig_r025"
MANIFEST_DIR = DATA / "frame_manifest"

# The 2026-09-05 package includes the audited YOLO11n checkpoints.
YOLO_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "yolo"
EYE_YOLO = YOLO_WEIGHTS / "e_best.pt"
INST_YOLO = YOLO_WEIGHTS / "i_best.pt"

OUT_ROOT = ROOT / "data/sics155_100_yolo_b2_multiinst_assets"


def prepare_out():
    for sub in ["npz", "meta", "vis", "groundTruth", "splits"]:
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


def read_manifest(video):
    p = MANIFEST_DIR / f"{video}.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def get_top1(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0, 0

    xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

    order = np.argsort(-conf)
    j = int(order[0])
    return xyxy[j], float(conf[j]), int(len(conf))


def get_topk(result, k=2):
    boxes = np.zeros((k, 4), dtype=np.float32)
    confs = np.zeros((k,), dtype=np.float32)
    valids = np.zeros((k,), dtype=np.float32)

    if result.boxes is None or len(result.boxes) == 0:
        return boxes, confs, valids, 0

    xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

    order = np.argsort(-conf)
    n = min(k, len(order))

    for out_i in range(n):
        j = int(order[out_i])
        boxes[out_i] = xyxy[j]
        confs[out_i] = float(conf[j])
        valids[out_i] = 1.0

    return boxes, confs, valids, int(len(conf))


def draw_box(img, box, conf, label, color):
    if box is None:
        return

    h, w = img.shape[:2]
    x1, y1, x2, y2 = box.astype(int).tolist()

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        f"{label} {conf:.2f}",
        (x1, max(22, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def process_video(video, eye_model, inst_model, args):
    npz_dir = OUT_ROOT / "npz"
    meta_dir = OUT_ROOT / "meta"
    vis_dir = OUT_ROOT / "vis" / video

    npz_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    out_npz = npz_dir / f"{video}.npz"
    out_meta = meta_dir / f"{video}_yolo_b2_multiinst.json"

    frame_paths = read_manifest(video)
    T = len(frame_paths)
    T_gt = gt_len(video)

    if T != T_gt:
        raise RuntimeError(f"{video}: manifest T={T}, gt T={T_gt}")

    if out_npz.exists() and not args.overwrite:
        z = np.load(out_npz)
        if "frame_ids" in z and len(z["frame_ids"]) == T and "inst_xyxy_multi" in z:
            print(f"[SKIP] {video}: exists, T={T}", flush=True)
            return {
                "video": video,
                "status": "skip",
                "T": int(T),
                "npz": str(out_npz),
            }

    frame_ids = np.arange(T, dtype=np.int32)

    eye_xyxy = np.zeros((T, 4), dtype=np.float32)
    eye_conf = np.zeros((T,), dtype=np.float32)
    eye_valid = np.zeros((T,), dtype=np.float32)
    eye_count = np.zeros((T,), dtype=np.float32)

    inst_xyxy = np.zeros((T, 4), dtype=np.float32)
    inst_conf = np.zeros((T,), dtype=np.float32)
    inst_valid = np.zeros((T,), dtype=np.float32)
    inst_count = np.zeros((T,), dtype=np.float32)

    max_inst = args.max_inst
    inst_xyxy_multi = np.zeros((T, max_inst, 4), dtype=np.float32)
    inst_conf_multi = np.zeros((T, max_inst), dtype=np.float32)
    inst_valid_multi = np.zeros((T, max_inst), dtype=np.float32)

    frame_widths = np.zeros((T,), dtype=np.int32)
    frame_heights = np.zeros((T,), dtype=np.int32)

    frames = []
    local_ids = []
    saved_vis = 0

    def flush_batch():
        nonlocal frames, local_ids, saved_vis

        if not frames:
            return

        eye_results = eye_model.predict(
            frames,
            imgsz=args.imgsz,
            conf=args.eye_conf,
            iou=args.iou,
            max_det=5,
            device=args.device,
            verbose=False,
        )

        inst_results = inst_model.predict(
            frames,
            imgsz=args.imgsz,
            conf=args.inst_conf,
            iou=args.iou,
            max_det=max_inst,
            device=args.device,
            verbose=False,
        )

        for frame, local_t, er, ir in zip(frames, local_ids, eye_results, inst_results):
            H, W = frame.shape[:2]
            frame_heights[local_t] = H
            frame_widths[local_t] = W

            e_box, e_conf, e_count = get_top1(er)
            eye_count[local_t] = e_count
            if e_box is not None:
                eye_xyxy[local_t] = e_box
                eye_conf[local_t] = e_conf
                eye_valid[local_t] = 1.0

            i_boxes, i_confs, i_valids, i_count = get_topk(ir, k=max_inst)
            inst_xyxy_multi[local_t] = i_boxes
            inst_conf_multi[local_t] = i_confs
            inst_valid_multi[local_t] = i_valids
            inst_count[local_t] = i_count

            if i_valids[0] > 0:
                inst_xyxy[local_t] = i_boxes[0]
                inst_conf[local_t] = i_confs[0]
                inst_valid[local_t] = 1.0

            if args.save_vis and (local_t % args.vis_stride == 0):
                vis = frame.copy()
                if eye_valid[local_t] > 0:
                    draw_box(vis, eye_xyxy[local_t], eye_conf[local_t], "eye", (0, 255, 0))
                for k in range(max_inst):
                    if inst_valid_multi[local_t, k] > 0:
                        draw_box(
                            vis,
                            inst_xyxy_multi[local_t, k],
                            inst_conf_multi[local_t, k],
                            f"inst{k+1}",
                            (0, 0, 255),
                        )

                src_path = Path(frame_paths[local_t])
                phase = src_path.parent.name
                stem = f"t{local_t:06d}_{phase}_{src_path.stem}.jpg"
                cv2.imwrite(str(vis_dir / stem), vis)
                saved_vis += 1

        frames = []
        local_ids = []

    for t, fp in enumerate(frame_paths):
        frame = cv2.imread(fp)
        if frame is None:
            raise RuntimeError(f"{video}: cannot read frame {t}: {fp}")

        frames.append(frame)
        local_ids.append(t)

        if len(frames) >= args.batch:
            flush_batch()

        if args.progress_every > 0 and (t + 1) % args.progress_every == 0:
            print(f"[PROGRESS] {video}: {t+1}/{T}", flush=True)

    flush_batch()

    width = int(frame_widths[frame_widths > 0][0]) if np.any(frame_widths > 0) else 0
    height = int(frame_heights[frame_heights > 0][0]) if np.any(frame_heights > 0) else 0

    np.savez_compressed(
        out_npz,
        frame_ids=frame_ids,
        frame_paths=np.asarray(frame_paths),
        width=np.array(width, dtype=np.int32),
        height=np.array(height, dtype=np.int32),
        frame_widths=frame_widths,
        frame_heights=frame_heights,
        start_frame=np.array(0, dtype=np.int32),

        eye_xyxy=eye_xyxy,
        eye_conf=eye_conf,
        eye_valid=eye_valid,
        eye_count=eye_count,

        inst_xyxy=inst_xyxy,
        inst_conf=inst_conf,
        inst_valid=inst_valid,
        inst_count=inst_count,

        inst_xyxy_multi=inst_xyxy_multi,
        inst_conf_multi=inst_conf_multi,
        inst_valid_multi=inst_valid_multi,
    )

    meta = {
        "video": video,
        "T": int(T),
        "width": int(width),
        "height": int(height),
        "eye_valid_ratio": float(eye_valid.mean()),
        "inst_valid_ratio": float(inst_valid.mean()),
        "inst_count_mean": float(inst_count.mean()),
        "eye_conf_mean_valid": float(eye_conf[eye_valid > 0].mean()) if np.any(eye_valid > 0) else 0.0,
        "inst_conf_mean_valid": float(inst_conf[inst_valid > 0].mean()) if np.any(inst_valid > 0) else 0.0,
        "saved_vis": int(saved_vis),
        "npz": str(out_npz),
    }

    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[OK] {video}: T={T}, eye_valid={meta['eye_valid_ratio']:.4f}, "
        f"inst_valid={meta['inst_valid_ratio']:.4f}, saved_vis={saved_vis}",
        flush=True,
    )

    return {
        "video": video,
        "status": "ok",
        **meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="all", help="all or split bundle name, e.g. train.split1.bundle")
    ap.add_argument("--video", default=None, help="single video id, e.g. BL_0101")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eye_yolo", default=str(EYE_YOLO))
    ap.add_argument("--inst_yolo", default=str(INST_YOLO))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--eye_conf", type=float, default=0.25)
    ap.add_argument("--inst_conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_inst", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--save_vis", type=int, default=0)
    ap.add_argument("--vis_stride", type=int, default=3000)
    ap.add_argument("--progress_every", type=int, default=1000)
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
    print("[SICS155_100 YOLO B2 MULTI-INST ASSET EXTRACTION]", flush=True)
    print("DATA:", DATA, flush=True)
    print("MANIFEST:", MANIFEST_DIR, flush=True)
    print("OUT_ROOT:", OUT_ROOT, flush=True)
    print("eye_yolo:", args.eye_yolo, flush=True)
    print("inst_yolo:", args.inst_yolo, flush=True)
    print("bundle:", args.bundle, flush=True)
    print("videos:", len(videos), videos[:10], "...", flush=True)
    print("batch:", args.batch, "imgsz:", args.imgsz, "device:", args.device, flush=True)
    print("=" * 80, flush=True)

    eye_model = YOLO(args.eye_yolo)
    inst_model = YOLO(args.inst_yolo)

    summary_path = OUT_ROOT / "meta" / "summary_all.jsonl"
    if args.overwrite and summary_path.exists():
        summary_path.unlink()

    for i, video in enumerate(videos, start=1):
        print(f"\n===== [{i}/{len(videos)}] {video} =====", flush=True)
        try:
            res = process_video(video, eye_model, inst_model, args)
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
    print("Output:", OUT_ROOT, flush=True)
    print("Summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()

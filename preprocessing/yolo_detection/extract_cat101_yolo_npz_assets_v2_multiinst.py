import os
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))
CAT = Path(os.environ.get("CATARACT101_ROOT", "/path/to/cataract-101"))
A0 = ROOT / "data/cataract101_m10_a0_rgb2048"

EYE_YOLO = ROOT / "m10/e_best.pt"
INST_YOLO = ROOT / "m10/i_best.pt"

OUT_ROOT = ROOT / "data/cataract101_m10_yolo_b2_multiinst_assets"


def first_annotation_map():
    first = {}
    with open(CAT / "annotations.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            vid = int(r["VideoID"])
            fr = int(r["FrameNo"])
            first[vid] = min(first.get(vid, fr), fr)
    return first


def read_ids(bundle):
    ids = []
    with open(A0 / "splits" / bundle, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            m = re.match(r"case_(\d+)\.txt", s)
            if m:
                ids.append(int(m.group(1)))
    return ids


def gt_len(vid):
    p = A0 / "groundTruth" / f"case_{vid}.txt"
    return sum(1 for _ in open(p, encoding="utf-8"))


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


def process_video(vid, eye_model, inst_model, first_map, args):
    npz_dir = OUT_ROOT / "npz"
    meta_dir = OUT_ROOT / "meta"
    vis_dir = OUT_ROOT / "vis" / f"case_{vid}"

    npz_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    out_npz = npz_dir / f"case_{vid}.npz"
    out_meta = meta_dir / f"case_{vid}_yolo_b2_multiinst.json"

    T = gt_len(vid)
    start = first_map[vid]

    if out_npz.exists() and not args.overwrite:
        z = np.load(out_npz)
        if "frame_ids" in z and len(z["frame_ids"]) == T and "inst_xyxy_multi" in z:
            print(f"[SKIP] case_{vid}: exists, T={T}")
            return

    video_path = CAT / "videos" / f"case_{vid}.mp4"
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_ids = np.arange(start, start + T, dtype=np.int32)

    eye_xyxy = np.zeros((T, 4), dtype=np.float32)
    eye_conf = np.zeros((T,), dtype=np.float32)
    eye_valid = np.zeros((T,), dtype=np.float32)
    eye_count = np.zeros((T,), dtype=np.float32)

    # top-1 compatibility fields, for geom20 simple version
    inst_xyxy = np.zeros((T, 4), dtype=np.float32)
    inst_conf = np.zeros((T,), dtype=np.float32)
    inst_valid = np.zeros((T,), dtype=np.float32)
    inst_count = np.zeros((T,), dtype=np.float32)

    # multi-instrument fields, max 2 instruments per frame
    max_inst = args.max_inst
    inst_xyxy_multi = np.zeros((T, max_inst, 4), dtype=np.float32)
    inst_conf_multi = np.zeros((T, max_inst), dtype=np.float32)
    inst_valid_multi = np.zeros((T, max_inst), dtype=np.float32)

    frames = []
    global_ids = []
    idx = 0
    used = 0
    width = None
    height = None
    saved_vis = 0

    def flush_batch():
        nonlocal frames, global_ids, saved_vis

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
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )

        for frame, gid, er, ir in zip(frames, global_ids, eye_results, inst_results):
            j = gid - start

            eb, ec, ecount = get_top1(er)
            if eb is not None:
                eye_xyxy[j] = eb
                eye_conf[j] = ec
                eye_valid[j] = 1.0
                eye_count[j] = ecount

            iboxes, iconfs, ivalids, icount = get_topk(ir, k=max_inst)
            inst_xyxy_multi[j] = iboxes
            inst_conf_multi[j] = iconfs
            inst_valid_multi[j] = ivalids
            inst_count[j] = icount

            if ivalids[0] > 0:
                inst_xyxy[j] = iboxes[0]
                inst_conf[j] = iconfs[0]
                inst_valid[j] = 1.0

            if args.save_vis and ((gid - start) % args.vis_stride == 0):
                vis = frame.copy()

                draw_box(vis, eb, ec, "eye", (0, 255, 0))

                colors = [(0, 0, 255), (255, 0, 255)]
                for kk in range(max_inst):
                    if ivalids[kk] > 0:
                        draw_box(
                            vis,
                            iboxes[kk],
                            iconfs[kk],
                            f"inst{kk + 1}",
                            colors[kk % len(colors)],
                        )

                title = (
                    f"case_{vid} frame={gid} "
                    f"eye={int(eye_valid[j])} inst_count={int(icount)} kept={int(ivalids.sum())}"
                )

                cv2.putText(
                    vis,
                    title,
                    (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    3,
                )
                cv2.putText(
                    vis,
                    title,
                    (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 0, 0),
                    1,
                )

                cv2.imwrite(str(vis_dir / f"frame_{gid:06d}.jpg"), vis)
                saved_vis += 1

        frames = []
        global_ids = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if width is None:
            height, width = frame.shape[:2]

        if idx >= start and used < T:
            frames.append(frame)
            global_ids.append(idx)
            used += 1

            if len(frames) >= args.batch:
                flush_batch()

        idx += 1

        if used >= T:
            break

    flush_batch()
    cap.release()

    if used != T:
        raise RuntimeError(f"case_{vid}: used={used}, expected T={T}")

    np.savez_compressed(
        out_npz,
        frame_ids=frame_ids,
        width=np.array(width, dtype=np.int32),
        height=np.array(height, dtype=np.int32),
        start_frame=np.array(start, dtype=np.int32),

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
        max_inst=np.array(max_inst, dtype=np.int32),
    )

    meta = {
        "video_id": int(vid),
        "video_path": str(video_path),
        "T": int(T),
        "start_frame": int(start),
        "width": int(width),
        "height": int(height),
        "eye_yolo": str(EYE_YOLO),
        "inst_yolo": str(INST_YOLO),
        "eye_conf_thres": float(args.eye_conf),
        "inst_conf_thres": float(args.inst_conf),
        "iou": float(args.iou),
        "max_inst": int(max_inst),
        "max_det": int(args.max_det),
        "eye_valid_ratio": float(eye_valid.mean()),
        "inst_valid_ratio_top1": float(inst_valid.mean()),
        "inst_valid_ratio_any": float((inst_valid_multi.max(axis=1) > 0).mean()),
        "inst_valid_ratio_second": float(inst_valid_multi[:, 1].mean()) if max_inst >= 2 else 0.0,
        "mean_eye_conf": float(eye_conf[eye_valid > 0].mean()) if eye_valid.sum() > 0 else 0.0,
        "mean_inst_conf_top1": float(inst_conf[inst_valid > 0].mean()) if inst_valid.sum() > 0 else 0.0,
        "mean_inst_count_raw": float(inst_count.mean()),
        "saved_vis": int(saved_vis),
        "npz": str(out_npz),
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] case_{vid}: "
        f"T={T}, "
        f"eye={meta['eye_valid_ratio']:.3f}, "
        f"inst_any={meta['inst_valid_ratio_any']:.3f}, "
        f"inst2={meta['inst_valid_ratio_second']:.3f}, "
        f"raw_count={meta['mean_inst_count_raw']:.3f}, "
        f"vis={saved_vis}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--video_id", type=int, default=None)
    ap.add_argument("--bundle", default="all.bundle")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)

    ap.add_argument("--eye_conf", type=float, default=0.15)
    ap.add_argument("--inst_conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--max_det", type=int, default=10)
    ap.add_argument("--max_inst", type=int, default=2)

    ap.add_argument("--device", default="0")
    ap.add_argument("--save_vis", type=int, default=1)
    ap.add_argument("--vis_stride", type=int, default=800)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    print("===== Cataract101 YOLO B2 Multi-Instrument Extraction =====")
    print("eye_yolo :", EYE_YOLO)
    print("inst_yolo:", INST_YOLO)
    print("out_root :", OUT_ROOT)
    print("max_inst :", args.max_inst)
    print("eye_conf :", args.eye_conf)
    print("inst_conf:", args.inst_conf)
    print("iou      :", args.iou)
    print("max_det  :", args.max_det)

    if not EYE_YOLO.exists():
        raise FileNotFoundError(EYE_YOLO)
    if not INST_YOLO.exists():
        raise FileNotFoundError(INST_YOLO)

    first_map = first_annotation_map()
    vids = [args.video_id] if args.video_id is not None else read_ids(args.bundle)

    eye_model = YOLO(str(EYE_YOLO))
    inst_model = YOLO(str(INST_YOLO))

    print("num videos:", len(vids))

    for i, vid in enumerate(vids, 1):
        print(f"\n===== [{i}/{len(vids)}] case_{vid} =====")
        process_video(vid, eye_model, inst_model, first_map, args)


if __name__ == "__main__":
    main()

import os
import argparse, csv, re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))
CAT = Path(os.environ.get("CATARACT101_ROOT", "/path/to/cataract-101"))

EYE_W = ROOT / "e_best.pt"
INST_W = ROOT / "i_best.pt"
OUT_ROOT = ROOT / "artifacts/m10_a1_cat101_yolo_vis"


def first_annotation_map():
    first = {}
    with open(CAT / "annotations.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            vid, fr = int(r["VideoID"]), int(r["FrameNo"])
            first[vid] = min(first.get(vid, fr), fr)
    return first


def get_phase_name(vid, global_frame):
    ann = []
    with open(CAT / "annotations.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if int(r["VideoID"]) == vid:
                ann.append((int(r["FrameNo"]), int(r["Phase"])))

    phase_map = {}
    with open(CAT / "phases.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter=";"):
            phase_map[int(r["Phase"])] = r["Meaning"]

    ann = sorted(ann)
    cur = "unlabeled"
    for fr, ph in ann:
        if global_frame >= fr:
            cur = phase_map.get(ph, str(ph))
        else:
            break
    return cur


def draw_boxes(img, result, name, color):
    lines = []
    if result.boxes is None or len(result.boxes) == 0:
        return lines

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    conf = result.boxes.conf.detach().cpu().numpy()

    h, w = img.shape[:2]

    for box, cf in zip(xyxy, conf):
        x1, y1, x2, y2 = box.astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = f"{name} {cf:.2f}"
        cv2.putText(img, txt, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{name} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {float(cf):.6f}")

    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", type=int, default=269)
    ap.add_argument("--stride", type=int, default=500)
    ap.add_argument("--max_save", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    vid = args.video_id
    first_map = first_annotation_map()
    start = first_map[vid]

    video_path = CAT / "videos" / f"case_{vid}.mp4"
    out_dir = OUT_ROOT / f"case_{vid}"
    img_dir = out_dir / "images"
    lab_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    eye_model = YOLO(str(EYE_W))
    inst_model = YOLO(str(INST_W))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    rows = []
    saved = 0
    idx = 0

    print(f"[INFO] video={video_path}")
    print(f"[INFO] first_annotation_frame={start}")
    print(f"[INFO] stride={args.stride}, max_save={args.max_save}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx >= start and ((idx - start) % args.stride == 0):
            eye_res = eye_model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
            inst_res = inst_model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]

            vis = frame.copy()
            eye_lines = draw_boxes(vis, eye_res, "eye", (0, 255, 0))
            inst_lines = draw_boxes(vis, inst_res, "instrument", (0, 0, 255))

            phase = get_phase_name(vid, idx)
            title = f"case_{vid} | frame={idx} | phase={phase}"
            cv2.putText(vis, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(vis, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

            stem = f"frame_{idx:06d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), vis)

            with open(lab_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                for line in eye_lines + inst_lines:
                    f.write(line + "\n")

            rows.append({
                "video_id": vid,
                "frame": idx,
                "phase": phase,
                "eye_count": len(eye_lines),
                "inst_count": len(inst_lines),
                "image": str(img_dir / f"{stem}.jpg"),
                "label": str(lab_dir / f"{stem}.txt"),
            })

            print(f"[SAVE] {stem} | phase={phase} | eye={len(eye_lines)} | inst={len(inst_lines)}")
            saved += 1

            if saved >= args.max_save:
                break

        idx += 1

    cap.release()

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "frame", "phase", "eye_count", "inst_count", "image", "label"])
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] saved={saved}")
    print(f"[OUT] {out_dir}")
    print(f"[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()

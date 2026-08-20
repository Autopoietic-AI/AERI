import os
import argparse
import shutil
import subprocess
import re
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ROOT", "/path/to/MS-TCN2"))
    ap.add_argument("--split", type=int, default=1)

    ap.add_argument("--base_dataset", default="sics155_100_a5_tig_stagefusion_r025_rsens_seed2026_trainval")
    ap.add_argument("--sf_dataset", default="sics155_100_a5_tig_stagefusion_r025_rsens_seed2026_trainval_smooth_v2_2")
    ap.add_argument("--vo_dataset", default="sics155_100_a5_visualonly_seed2026_trainval_smooth_v2_2")

    ap.add_argument("--out_dataset", default="sics155_100_a5_stagefusion_smooth2_vofallback_quality_trainval")

    ap.add_argument("--a4_dir", default="/path/to/MS-TCN2/data/sics155_100_a4_geo_assets_sam2_yolo_tiptrack_cons_compat/npz")
    ap.add_argument("--maskshape_dir", default="/path/to/MS-TCN2/data/sics155_100_a5_maskshape_assets_v1_safe/npz")

    ap.add_argument("--eye_thr", type=float, default=0.30)
    ap.add_argument("--mask_thr", type=float, default=0.10)

    # Optional prediction self-check. Default off for the clean paper-safe version.
    ap.add_argument("--use_frag", type=int, default=0)
    ap.add_argument("--pred_med_thr", type=float, default=2.0)
    ap.add_argument("--pred_seg_per_1k_thr", type=float, default=10.0)

    return ap.parse_args()


def parse_valid_labels(mapping):
    valid = set()
    for line in mapping.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0].isdigit():
            valid.add(" ".join(parts[1:]))
        elif parts[-1].isdigit():
            valid.add(" ".join(parts[:-1]))
    return valid


def read_pred(p, valid):
    return [x for x in p.read_text(errors="ignore").split() if x in valid]


def segments(seq):
    if not seq:
        return []
    out = []
    s = 0
    cur = seq[0]
    for i, x in enumerate(seq[1:], 1):
        if x != cur:
            out.append((cur, s, i))
            s = i
            cur = x
    out.append((cur, s, len(seq)))
    return out


def find_pred(pred_dir, video):
    stem = Path(video).stem
    candidates = [
        pred_dir / stem,
        pred_dir / f"{stem}.txt",
        pred_dir / video,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find prediction for {video} in {pred_dir}")


def load_quality(video, a4_dir, maskshape_dir):
    stem = Path(video).stem

    eye_mean = float("nan")
    mask_mean = float("nan")

    a4p = a4_dir / f"{stem}.npz"
    if a4p.exists():
        z = np.load(a4p)
        if "eye_valid" in z:
            eye_mean = float(np.asarray(z["eye_valid"]).mean())

    mp = maskshape_dir / f"{stem}.npz"
    if mp.exists():
        z = np.load(mp)
        if "mask_shape_valid" in z:
            mask_mean = float(np.asarray(z["mask_shape_valid"]).mean())

    return eye_mean, mask_mean


def eval_dataset(root, dataset, split):
    r = subprocess.run(
        ["python", str(root / "eval.py"), "--dataset", dataset, "--split", str(split)],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(r.stdout)

    print("===== FALLBACK RESULT =====")
    for k in ["Acc", "Edit", "F1@0.10", "F1@0.25", "F1@0.50"]:
        m = re.search(rf"{re.escape(k)}:\s*([0-9.]+)", r.stdout)
        if m:
            print(f"{k}: {float(m.group(1)):.4f}")


def main():
    args = parse_args()

    ROOT = Path(args.root)
    SPLIT = args.split

    BASE_DATA = ROOT / "data" / args.base_dataset
    SF_DATA = ROOT / "data" / args.sf_dataset
    VO_DATA = ROOT / "data" / args.vo_dataset

    SF_PRED = ROOT / "results" / f"tecno_{args.sf_dataset}" / f"split_{SPLIT}"
    VO_PRED = ROOT / "results" / f"tecno_{args.vo_dataset}" / f"split_{SPLIT}"

    OUT_DATA = ROOT / "data" / args.out_dataset
    OUT_PRED = ROOT / "results" / f"tecno_{args.out_dataset}" / f"split_{SPLIT}"

    a4_dir = Path(args.a4_dir)
    maskshape_dir = Path(args.maskshape_dir)

    valid = parse_valid_labels(BASE_DATA / "mapping.txt")

    test_bundle = BASE_DATA / "splits" / f"test.split{SPLIT}.bundle"
    videos = [x.strip() for x in test_bundle.read_text().splitlines() if x.strip()]

    print("============================================================")
    print("[SICS fallback eval]")
    print("base_dataset:", args.base_dataset)
    print("sf_dataset:", args.sf_dataset)
    print("vo_dataset:", args.vo_dataset)
    print("out_dataset:", args.out_dataset)
    print("split:", SPLIT)
    print("eye_thr:", args.eye_thr)
    print("mask_thr:", args.mask_thr)
    print("use_frag:", args.use_frag)
    print("videos:", len(videos))
    print("============================================================")

    if OUT_DATA.exists() or OUT_DATA.is_symlink():
        OUT_DATA.unlink() if OUT_DATA.is_symlink() else shutil.rmtree(OUT_DATA)
    OUT_DATA.mkdir(parents=True)

    shutil.copy(BASE_DATA / "mapping.txt", OUT_DATA / "mapping.txt")

    for name in ["groundTruth", "splits", "features", "geo_assets"]:
        src = BASE_DATA / name
        if src.exists():
            (OUT_DATA / name).symlink_to(src.resolve())

    if OUT_PRED.exists():
        shutil.rmtree(OUT_PRED)
    OUT_PRED.mkdir(parents=True)

    chosen_vo = []
    chosen_sf = []

    for video in videos:
        stem = Path(video).stem

        sf_p = find_pred(SF_PRED, video)
        vo_p = find_pred(VO_PRED, video)

        sf_seq = read_pred(sf_p, valid)
        sf_segs = segments(sf_seq)
        sf_seg_lens = [e - s for _, s, e in sf_segs]

        pred_med = float(np.median(sf_seg_lens)) if sf_seg_lens else 0.0
        pred_seg_per_1k = len(sf_segs) / max(len(sf_seq), 1) * 1000

        eye_mean, mask_mean = load_quality(video, a4_dir, maskshape_dir)

        quality_bad = (
            (not np.isnan(eye_mean) and eye_mean < args.eye_thr)
            or (not np.isnan(mask_mean) and mask_mean < args.mask_thr)
        )

        frag_bad = False
        if args.use_frag:
            frag_bad = (
                pred_med <= args.pred_med_thr
                or pred_seg_per_1k > args.pred_seg_per_1k_thr
            )

        use_vo = quality_bad or frag_bad

        if use_vo:
            chosen_vo.append((stem, eye_mean, mask_mean, pred_med, pred_seg_per_1k, quality_bad, frag_bad))
            shutil.copy(vo_p, OUT_PRED / sf_p.name)
        else:
            chosen_sf.append((stem, eye_mean, mask_mean, pred_med, pred_seg_per_1k, quality_bad, frag_bad))
            shutil.copy(sf_p, OUT_PRED / sf_p.name)

    EVAL_PARENT = ROOT / "results" / args.out_dataset
    EVAL_DIR = EVAL_PARENT / f"split_{SPLIT}"

    EVAL_PARENT.mkdir(parents=True, exist_ok=True)
    if EVAL_DIR.exists() or EVAL_DIR.is_symlink():
        EVAL_DIR.unlink() if EVAL_DIR.is_symlink() else shutil.rmtree(EVAL_DIR)
    EVAL_DIR.symlink_to(OUT_PRED.resolve())

    print()
    print("===== fallback to visual-only videos =====")
    if not chosen_vo:
        print("(none)")
    for x in chosen_vo:
        stem, eye, mask, pred_med, seg1k, qbad, fbad = x
        print(
            f"{stem:8s} eye={eye:.4f} mask={mask:.4f} "
            f"pred_med={pred_med:.1f} seg_per_1k={seg1k:.2f} "
            f"quality_bad={qbad} frag_bad={fbad}"
        )

    print()
    print("chosen visual-only:", len(chosen_vo))
    print("chosen StageFusion:", len(chosen_sf))

    print()
    eval_dataset(ROOT, args.out_dataset, SPLIT)


if __name__ == "__main__":
    main()

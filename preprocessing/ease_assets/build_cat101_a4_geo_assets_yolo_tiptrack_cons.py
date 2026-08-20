# -*- coding: utf-8 -*-
import argparse
import importlib.util
import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))
SAM2 = ROOT / "data/cataract101_m10_a2c_sam2_masks_full/npz"
YOLO = ROOT / "data/cataract101_m10_yolo_b2_multiinst_assets/npz"
DEBUG_PY = ROOT / "m10/a4_geoig/debug_cat101_a4_v1c_cons_yolo_tiptrack.py"

OUT_ROOT = ROOT / "data/cataract101_m10_a4_geo_assets_sam2_yolo_tiptrack_cons"
OUT_NPZ = OUT_ROOT / "npz"
OUT_META = OUT_ROOT / "meta"


def load_debug_module():
    spec = importlib.util.spec_from_file_location("v1c_cons", str(DEBUG_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_case_ids():
    sam_cases = {p.stem.replace("case_", "") for p in SAM2.glob("case_*.npz")}
    yolo_cases = {p.stem.replace("case_", "") for p in YOLO.glob("case_*.npz")}
    common = sorted(sam_cases & yolo_cases, key=lambda x: int(x))
    return [int(x) for x in common]


def convert_debug_npz_to_standard(src_npz, dst_npz):
    z = np.load(src_npz, allow_pickle=True)

    save_dict = {}
    for k in z.files:
        save_dict[k] = z[k]

    # 保持和之前 A4 asset 兼容的核心 key
    required = ["frame_ids", "eye_center", "eye_radius", "inst_tips",
                "inst_centers", "presence_mask", "geo", "traj"]

    missing = [k for k in required if k not in save_dict]
    if missing:
        raise RuntimeError(f"Missing required keys {missing} in {src_npz}")

    np.savez_compressed(dst_npz, **save_dict)


def worker(args_tuple):
    vid, conf_th, box_expand, min_area, force = args_tuple

    out_case_dir = OUT_META / f"case_{vid}"
    out_case_dir.mkdir(parents=True, exist_ok=True)

    dst_npz = OUT_NPZ / f"case_{vid}.npz"
    dst_summary = OUT_META / f"case_{vid}_summary.json"

    if dst_npz.exists() and dst_summary.exists() and not force:
        return {
            "video_id": vid,
            "status": "skip",
            "npz": str(dst_npz)
        }

    try:
        mod = load_debug_module()

        debug_npz = mod.build_tiptrack(
            video_id=vid,
            out_dir=out_case_dir,
            conf_th=conf_th,
            box_expand=box_expand,
            min_area=min_area
        )

        # Fast path: debug npz already contains required keys.
        # Copy directly to avoid second load + second npz compression.
        shutil.copyfile(debug_npz, dst_npz)
        try:
            Path(debug_npz).unlink()
        except Exception:
            pass

        # 找 debug summary
        debug_summary = out_case_dir / f"case_{vid}_v1c_cons_yolo_tiptrack_summary.json"
        if debug_summary.exists():
            shutil.copyfile(debug_summary, dst_summary)
            summary = json.loads(debug_summary.read_text(encoding="utf-8"))
        else:
            summary = {"video_id": vid}

        summary["status"] = "ok"
        summary["standard_npz"] = str(dst_npz)

        return summary

    except Exception as e:
        return {
            "video_id": vid,
            "status": "error",
            "error": repr(e)
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conf_th", type=float, default=0.30)
    ap.add_argument("--box_expand", type=float, default=0.08)
    ap.add_argument("--min_area", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N cases")
    args = ap.parse_args()

    OUT_NPZ.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    case_ids = get_case_ids()
    if args.limit and args.limit > 0:
        case_ids = case_ids[:args.limit]

    print("=" * 80, flush=True)
    print("[A4-v1c-cons full asset builder]", flush=True)
    print("SAM2:", SAM2, flush=True)
    print("YOLO:", YOLO, flush=True)
    print("OUT_NPZ:", OUT_NPZ, flush=True)
    print("cases:", len(case_ids), case_ids[:10], "...", flush=True)
    print("workers:", args.workers, flush=True)
    print("conf_th:", args.conf_th, flush=True)
    print("box_expand:", args.box_expand, flush=True)
    print("min_area:", args.min_area, flush=True)
    print("force:", args.force, flush=True)
    print("=" * 80, flush=True)

    tasks = [
        (vid, args.conf_th, args.box_expand, args.min_area, args.force)
        for vid in case_ids
    ]

    summary_path = OUT_META / "summary_all.jsonl"

    if args.workers <= 1:
        results = []
        for t in tasks:
            res = worker(t)
            results.append(res)
            print(json.dumps(res, ensure_ascii=False), flush=True)
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
    else:
        with Pool(processes=args.workers) as pool:
            for res in pool.imap_unordered(worker, tasks):
                print(json.dumps(res, ensure_ascii=False), flush=True)
                with open(summary_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("[DONE]", flush=True)
    print("Output:", OUT_NPZ, flush=True)
    print("Summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()

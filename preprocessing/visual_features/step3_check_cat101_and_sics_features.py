import os
import csv
import numpy as np
from pathlib import Path

CAT_SRC = Path(os.environ.get("CATARACT101_ROOT", "/path/to/cataract-101"))
CAT_DATA = Path("/path/to/MS-TCN2/data/cataract101_m10_raw")

SICS_FEATURE_DIRS = [
    Path("/path/to/MS-TCN2/data/sics155_m8_v14a_attn_geom_full45/features"),
    Path("/path/to/MS-TCN2/data/sics155/features"),
    Path("/path/to/MS-TCN2/data/sics155_m7_v1_assets_norm"),
    Path("/path/to/MS-TCN2/data/sics155_m3_igv1_assets"),
]

print("===== Cataract101 GT length check =====")

videos = {}
with open(CAT_SRC / "videos.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f, delimiter=";"):
        videos[int(r["VideoID"])] = int(r["Frames"])

first = {}
with open(CAT_SRC / "annotations.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f, delimiter=";"):
        vid, fr = int(r["VideoID"]), int(r["FrameNo"])
        first[vid] = min(first.get(vid, fr), fr)

bad = []
for vid, total in videos.items():
    gt = CAT_DATA / "groundTruth" / f"case_{vid}.txt"
    n = sum(1 for _ in open(gt, encoding="utf-8"))
    expect = total - first[vid]
    if n != expect:
        bad.append((vid, n, expect))

print("checked videos:", len(videos))
print("bad count:", len(bad))
print("bad samples:", bad[:10])

print("\n===== Existing SICS feature format =====")
for d in SICS_FEATURE_DIRS:
    print("\nDIR:", d)
    if not d.exists():
        print("  not exists")
        continue

    files = sorted([p for p in d.glob("*") if p.suffix.lower() in [".npy", ".npz"]])
    print("  file count:", len(files))

    for p in files[:3]:
        if p.suffix == ".npy":
            arr = np.load(p, allow_pickle=True)
            print(" ", p.name, "shape:", arr.shape, "dtype:", arr.dtype)
        else:
            z = np.load(p, allow_pickle=True)
            print(" ", p.name, "keys:", list(z.keys()))
            for k in list(z.keys())[:6]:
                v = z[k]
                if hasattr(v, "shape"):
                    print("   ", k, v.shape, v.dtype)

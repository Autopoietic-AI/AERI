import os
import csv, json, shutil
from pathlib import Path
from collections import defaultdict, Counter

RAW = Path("/path/to/MS-TCN2/data/cataract101_m10_raw")
OUT = Path("/path/to/MS-TCN2/data/cataract101_m10_a0_rgb2048")
SRC = Path(os.environ.get("CATARACT101_ROOT", "/path/to/cataract-101"))

for sub in ["groundTruth", "splits", "features", "meta"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# copy mapping and GT
shutil.copy2(RAW / "mapping.txt", OUT / "mapping.txt")
for p in (RAW / "groundTruth").glob("case_*.txt"):
    shutil.copy2(p, OUT / "groundTruth" / p.name)

video_rows = []
by_surgeon = defaultdict(list)
by_exp = Counter()

with open(SRC / "videos.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f, delimiter=";"):
        vid = int(r["VideoID"])
        surgeon = int(r["Surgeon"])
        exp = int(r["Experience"])
        name = f"case_{vid}.txt"
        video_rows.append({
            "video_id": vid,
            "file": name,
            "frames": int(r["Frames"]),
            "fps": int(r["FPS"]),
            "surgeon": surgeon,
            "experience": exp,
        })
        by_surgeon[surgeon].append(name)
        by_exp[exp] += 1

all_files = sorted([x["file"] for x in video_rows])

def write_bundle(path, files):
    with open(path, "w", encoding="utf-8") as f:
        for x in sorted(files):
            f.write(x + "\n")

write_bundle(OUT / "splits" / "all.bundle", all_files)

fold_summary = {}
for fold, surgeon in enumerate(sorted(by_surgeon), start=1):
    test_files = sorted(by_surgeon[surgeon])
    train_files = sorted([x for x in all_files if x not in set(test_files)])
    write_bundle(OUT / "splits" / f"train.split{fold}.bundle", train_files)
    write_bundle(OUT / "splits" / f"test.split{fold}.bundle", test_files)
    fold_summary[f"split{fold}"] = {
        "heldout_surgeon": surgeon,
        "train_videos": len(train_files),
        "test_videos": len(test_files),
        "test_files": test_files[:10],
    }

with open(OUT / "meta" / "cat101_4fold_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "out": str(OUT),
        "fold_type": "leave-one-surgeon-out 4-fold",
        "total_videos": len(all_files),
        "surgeon_video_counts": {str(k): len(v) for k, v in sorted(by_surgeon.items())},
        "experience_video_counts": dict(by_exp),
        "folds": fold_summary,
    }, f, ensure_ascii=False, indent=2)

print(json.dumps({
    "out": str(OUT),
    "total_videos": len(all_files),
    "surgeon_video_counts": {str(k): len(v) for k, v in sorted(by_surgeon.items())},
    "folds": fold_summary,
}, ensure_ascii=False, indent=2))

import os
from pathlib import Path
import shutil
import subprocess
import re

ROOT = Path(os.environ.get("ROOT", "/path/to/MS-TCN2"))
SRC_DATASET = "sics155_100_a5_tig_stagefusion_r025_rsens_seed2026_trainval"
SPLIT = 1

SRC_PRED = ROOT / "results" / f"tecno_{SRC_DATASET}" / f"split_{SPLIT}"
GT_DATA = ROOT / "data" / SRC_DATASET

valid_labels = []
valid_set = set()

for line in (GT_DATA / "mapping.txt").read_text(errors="ignore").splitlines():
    parts = line.strip().split()
    if not parts:
        continue
    if parts[0].isdigit():
        lab = " ".join(parts[1:])
    elif parts[-1].isdigit():
        lab = " ".join(parts[:-1])
    else:
        continue
    valid_labels.append(lab)
    valid_set.add(lab)


def read_pred(p):
    toks = p.read_text(errors="ignore").split()
    return [x for x in toks if x in valid_set]


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


def smooth_preserve_len(seq, min_len, max_iter=20):
    seq = list(seq)
    n = len(seq)

    if min_len <= 1 or n == 0:
        return seq

    for _ in range(max_iter):
        segs = segments(seq)
        changed = False

        for idx, (lab, s, e) in enumerate(segs):
            length = e - s
            if length >= min_len:
                continue

            left = segs[idx - 1] if idx > 0 else None
            right = segs[idx + 1] if idx < len(segs) - 1 else None

            if left is None and right is None:
                continue
            elif left is None:
                new_lab = right[0]
            elif right is None:
                new_lab = left[0]
            else:
                left_len = left[2] - left[1]
                right_len = right[2] - right[1]
                new_lab = left[0] if left_len >= right_len else right[0]

            for j in range(s, e):
                seq[j] = new_lab

            changed = True

        if not changed:
            break

    assert len(seq) == n
    return seq


def make_dataset_alias(dataset):
    data_dir = ROOT / "data" / dataset
    if data_dir.exists() or data_dir.is_symlink():
        if data_dir.is_symlink():
            data_dir.unlink()
        else:
            shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    shutil.copy(GT_DATA / "mapping.txt", data_dir / "mapping.txt")

    for name in ["groundTruth", "splits", "features", "geo_assets"]:
        src = GT_DATA / name
        if src.exists():
            (data_dir / name).symlink_to(src.resolve())

    return data_dir


def eval_dataset(dataset):
    cmd = ["python", str(ROOT / "eval.py"), "--dataset", dataset, "--split", str(SPLIT)]
    r = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    text = r.stdout
    vals = {}
    for k in ["Acc", "Edit", "F1@0.10", "F1@0.25", "F1@0.50"]:
        m = re.search(rf"{re.escape(k)}:\s*([0-9.]+)", text)
        if m:
            vals[k] = float(m.group(1))

    if len(vals) < 5:
        print("\n[WARN] eval failed or incomplete for", dataset)
        print(text[-2000:])

    return vals


print("min_len,Acc,Edit,F1@10,F1@25,F1@50")

for min_len in [0, 2, 5, 10, 15, 25, 50, 75, 100, 150, 200, 300]:
    DATASET = f"{SRC_DATASET}_smooth_v2_{min_len}"
    DATA_DIR = make_dataset_alias(DATASET)

    OUT_DIR = ROOT / "results" / f"tecno_{DATASET}" / f"split_{SPLIT}"
    EVAL_PARENT = ROOT / "results" / DATASET
    EVAL_DIR = EVAL_PARENT / f"split_{SPLIT}"

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    for p in SRC_PRED.iterdir():
        if not p.is_file():
            continue

        seq = read_pred(p)
        sm = seq if min_len == 0 else smooth_preserve_len(seq, min_len)

        if len(seq) != len(sm):
            raise RuntimeError(f"length changed: {p.name} {len(seq)} -> {len(sm)}")

        (OUT_DIR / p.name).write_text(
            "### Frame level recognition: ###\n" + " ".join(sm) + "\n",
            encoding="utf-8"
        )

    EVAL_PARENT.mkdir(parents=True, exist_ok=True)

    if EVAL_DIR.exists() or EVAL_DIR.is_symlink():
        if EVAL_DIR.is_symlink():
            EVAL_DIR.unlink()
        else:
            shutil.rmtree(EVAL_DIR)

    EVAL_DIR.symlink_to(OUT_DIR.resolve())

    vals = eval_dataset(DATASET)

    print(
        f"{min_len},"
        f"{vals.get('Acc', float('nan')):.4f},"
        f"{vals.get('Edit', float('nan')):.4f},"
        f"{vals.get('F1@0.10', float('nan')):.4f},"
        f"{vals.get('F1@0.25', float('nan')):.4f},"
        f"{vals.get('F1@0.50', float('nan')):.4f}"
    )

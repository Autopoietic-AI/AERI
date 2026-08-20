import os
import argparse
import shutil
import subprocess
import re
from pathlib import Path

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

def smooth_preserve_len(seq, min_len, max_iter=20):
    seq = list(seq)
    n = len(seq)

    if min_len <= 1 or n == 0:
        return seq

    for _ in range(max_iter):
        segs = segments(seq)
        changed = False

        for idx, (lab, s, e) in enumerate(segs):
            if e - s >= min_len:
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

def find_result_dir(root, dataset, split):
    candidates = [
        root / "results" / f"tecno_{dataset}" / f"split_{split}",
        root / "results" / dataset / f"split_{split}",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find prediction dir for dataset={dataset}, split={split}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ROOT", "/path/to/MS-TCN2"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--suffix", default="smooth_v2_2")
    args = ap.parse_args()

    ROOT = Path(args.root)
    SRC_DATASET = args.dataset
    SPLIT = args.split
    OUT_DATASET = f"{SRC_DATASET}_{args.suffix}"

    SRC_DATA = ROOT / "data" / SRC_DATASET
    SRC_PRED = find_result_dir(ROOT, SRC_DATASET, SPLIT)

    OUT_DATA = ROOT / "data" / OUT_DATASET
    OUT_PRED = ROOT / "results" / f"tecno_{OUT_DATASET}" / f"split_{SPLIT}"

    EVAL_PARENT = ROOT / "results" / OUT_DATASET
    EVAL_DIR = EVAL_PARENT / f"split_{SPLIT}"

    valid = parse_valid_labels(SRC_DATA / "mapping.txt")

    # Rebuild data alias. This is safe across splits; prediction folders are kept per split.
    if OUT_DATA.exists() or OUT_DATA.is_symlink():
        if OUT_DATA.is_symlink():
            OUT_DATA.unlink()
        else:
            shutil.rmtree(OUT_DATA)

    OUT_DATA.mkdir(parents=True)
    shutil.copy(SRC_DATA / "mapping.txt", OUT_DATA / "mapping.txt")

    for name in ["groundTruth", "splits", "features", "geo_assets"]:
        src = SRC_DATA / name
        if src.exists():
            (OUT_DATA / name).symlink_to(src.resolve())

    if OUT_PRED.exists():
        shutil.rmtree(OUT_PRED)
    OUT_PRED.mkdir(parents=True)

    n_files = 0
    for p in SRC_PRED.iterdir():
        if not p.is_file():
            continue

        seq = read_pred(p, valid)
        sm = smooth_preserve_len(seq, args.min_len)

        if len(seq) != len(sm):
            raise RuntimeError(f"length changed: {p.name} {len(seq)} -> {len(sm)}")

        (OUT_PRED / p.name).write_text(
            "### Frame level recognition: ###\n" + " ".join(sm) + "\n",
            encoding="utf-8",
        )
        n_files += 1

    EVAL_PARENT.mkdir(parents=True, exist_ok=True)

    if EVAL_DIR.exists() or EVAL_DIR.is_symlink():
        if EVAL_DIR.is_symlink():
            EVAL_DIR.unlink()
        else:
            shutil.rmtree(EVAL_DIR)

    EVAL_DIR.symlink_to(OUT_PRED.resolve())

    print("============================================================")
    print("[Smooth eval]")
    print("SRC_DATASET:", SRC_DATASET)
    print("OUT_DATASET:", OUT_DATASET)
    print("SPLIT:", SPLIT)
    print("MIN_LEN:", args.min_len)
    print("SRC_PRED:", SRC_PRED)
    print("OUT_PRED:", OUT_PRED)
    print("files:", n_files)
    print("============================================================")

    r = subprocess.run(
        ["python", str(ROOT / "eval.py"), "--dataset", OUT_DATASET, "--split", str(SPLIT)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(r.stdout)

    print("===== SMOOTH RESULT =====")
    for k in ["Acc", "Edit", "F1@0.10", "F1@0.25", "F1@0.50"]:
        m = re.search(rf"{re.escape(k)}:\s*([0-9.]+)", r.stdout)
        if m:
            print(f"{k}: {float(m.group(1)):.4f}")

if __name__ == "__main__":
    main()

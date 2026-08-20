import os
from pathlib import Path
from collections import Counter
import argparse
import numpy as np
import csv

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ROOT", "/path/to/MS-TCN2"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", type=int, default=1)
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

def read_seq(p, valid):
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

def collapse(seq):
    return [x for x, _, _ in segments(seq)]

def edit_score(gt, pr):
    g, p = collapse(gt), collapse(pr)
    m, n = len(g), len(p)
    if max(m, n) == 0:
        return 100.0
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = 0 if g[i - 1] == p[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + c)
    return (1 - dp[m, n] / max(m, n)) * 100

def f1_counts(gt, pr, overlap):
    gs = segments(gt)
    ps = segments(pr)
    hits = set()
    tp = fp = 0

    for pl, ps0, pe0 in ps:
        best_iou = 0
        best_j = -1
        for j, (gl, gs0, ge0) in enumerate(gs):
            if j in hits or gl != pl:
                continue
            inter = max(0, min(pe0, ge0) - max(ps0, gs0))
            union = max(pe0, ge0) - min(ps0, gs0)
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= overlap:
            tp += 1
            hits.add(best_j)
        else:
            fp += 1

    fn = len(gs) - len(hits)
    return tp, fp, fn

def f1(tp, fp, fn):
    return 0 if 2 * tp + fp + fn == 0 else 100 * 2 * tp / (2 * tp + fp + fn)

def find_pred(pred_dir, video):
    stem = Path(video).stem
    for p in [pred_dir / stem, pred_dir / f"{stem}.txt", pred_dir / video]:
        if p.exists():
            return p
    raise FileNotFoundError(video)

def main():
    args = parse_args()
    ROOT = Path(args.root)
    DATASET = args.dataset
    SPLIT = args.split

    D = ROOT / "data" / DATASET
    G = D / "groundTruth"
    P = ROOT / "results" / f"tecno_{DATASET}" / f"split_{SPLIT}"
    S = D / "splits" / f"test.split{SPLIT}.bundle"
    OUT = ROOT / "diagnostics" / f"{DATASET}_split{SPLIT}_clean"
    OUT.mkdir(parents=True, exist_ok=True)

    valid = parse_valid_labels(D / "mapping.txt")
    videos = [x.strip() for x in S.read_text().splitlines() if x.strip()]

    total_correct = 0
    total_frames = 0
    edits = []
    f10 = [0, 0, 0]
    f25 = [0, 0, 0]
    f50 = [0, 0, 0]

    rows = []
    gt_counter = Counter()
    pr_counter = Counter()

    for v in videos:
        stem = Path(v).stem
        gt = read_seq(G / f"{stem}.txt", valid)
        pr = read_seq(find_pred(P, v), valid)

        L = min(len(gt), len(pr))
        gt = gt[:L]
        pr = pr[:L]

        gt_counter.update(gt)
        pr_counter.update(pr)

        correct = sum(a == b for a, b in zip(gt, pr))
        total_correct += correct
        total_frames += L

        gs = segments(gt)
        ps = segments(pr)
        e = edit_score(gt, pr)
        c10 = f1_counts(gt, pr, 0.10)
        c25 = f1_counts(gt, pr, 0.25)
        c50 = f1_counts(gt, pr, 0.50)

        for a, c in [(f10, c10), (f25, c25), (f50, c50)]:
            a[0] += c[0]
            a[1] += c[1]
            a[2] += c[2]

        rows.append({
            "video": stem,
            "T": L,
            "acc": 100 * correct / max(L, 1),
            "edit": e,
            "f1@10": f1(*c10),
            "f1@25": f1(*c25),
            "f1@50": f1(*c50),
            "gt_seg": len(gs),
            "pred_seg": len(ps),
            "seg_ratio": len(ps) / max(len(gs), 1),
            "pred_med_len": np.median([b - a for _, a, b in ps]) if ps else 0,
        })

    print("===== clean eval-like =====")
    print(f"Acc: {100 * total_correct / total_frames:.4f}")
    print(f"Edit: {np.mean([r['edit'] for r in rows]):.4f}")
    print(f"F1@0.10: {f1(*f10):.4f}")
    print(f"F1@0.25: {f1(*f25):.4f}")
    print(f"F1@0.50: {f1(*f50):.4f}")

    print("\n===== key videos =====")
    for k in ["SD_0307", "SD_0315", "SD_0321", "SD_0290", "BL_0168", "BL_0176"]:
        for r in rows:
            if r["video"] == k:
                print(
                    f"{k:8s} acc={r['acc']:.2f} edit={r['edit']:.2f} "
                    f"f25={r['f1@25']:.2f} f50={r['f1@50']:.2f} "
                    f"seg_ratio={r['seg_ratio']:.2f} pred_med={r['pred_med_len']:.1f}"
                )

    with (OUT / "per_video_clean.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n[SAVED]", OUT / "per_video_clean.csv")

if __name__ == "__main__":
    main()

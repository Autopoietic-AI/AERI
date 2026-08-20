import os
import argparse
import csv
import re
from pathlib import Path
from collections import Counter
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def read_seq(p: Path):
    # 核心修正：按 whitespace token 读，而不是按行读
    return [x.strip() for x in p.read_text(encoding="utf-8", errors="ignore").split() if x.strip()]


def parse_mapping(p: Path):
    id_to_label = {}
    label_to_id = {}

    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        if parts[0].isdigit():
            idx = int(parts[0])
            lab = " ".join(parts[1:])
        elif parts[-1].isdigit():
            idx = int(parts[-1])
            lab = " ".join(parts[:-1])
        else:
            continue

        id_to_label[idx] = lab
        label_to_id[lab] = idx

    return id_to_label, label_to_id


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


def edit_score(gt, pred):
    g = collapse(gt)
    p = collapse(pred)

    m, n = len(g), len(p)
    if max(m, n) == 0:
        return 100.0

    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if g[i - 1] == p[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )

    return (1.0 - dp[m, n] / max(m, n)) * 100.0


def segment_f1_counts(gt, pred, overlap):
    gt_segs = segments(gt)
    pr_segs = segments(pred)

    hits = set()
    tp = 0
    fp = 0

    for pl, ps, pe in pr_segs:
        best_iou = 0.0
        best_j = -1

        for j, (gl, gs, ge) in enumerate(gt_segs):
            if j in hits or gl != pl:
                continue

            inter = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= overlap:
            tp += 1
            hits.add(best_j)
        else:
            fp += 1

    fn = len(gt_segs) - len(hits)
    return tp, fp, fn


def f1_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 100.0 * (2 * tp) / denom


def safe_mean_npz(npz_path, key):
    if not npz_path.exists():
        return np.nan

    try:
        z = np.load(npz_path, allow_pickle=True)
        if key not in z:
            return np.nan
        arr = np.asarray(z[key])
        return float(np.nanmean(arr))
    except Exception:
        return np.nan


def corr(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 3:
        return np.nan
    if np.std(xs[m]) < 1e-8 or np.std(ys[m]) < 1e-8:
        return np.nan

    return float(np.corrcoef(xs[m], ys[m])[0, 1])


def find_pred(pred_dir, video):
    cands = [
        pred_dir / video,
        pred_dir / f"{video}.txt",
        pred_dir / Path(video).stem,
        pred_dir / f"{Path(video).stem}.txt",
    ]

    for p in cands:
        if p.exists() and p.is_file():
            return p

    return None


def find_gt(gt_dir, video):
    cands = [
        gt_dir / video,
        gt_dir / f"{video}.txt",
        gt_dir / Path(video).stem,
        gt_dir / f"{Path(video).stem}.txt",
    ]

    for p in cands:
        if p.exists() and p.is_file():
            return p

    return None


def to_ids(seq, label_to_id, id_to_label):
    ids = []

    for x in seq:
        if x not in label_to_id:
            new_id = max(label_to_id.values(), default=-1) + 1
            label_to_id[x] = new_id
            id_to_label[new_id] = x
        ids.append(label_to_id[x])

    return np.asarray(ids, dtype=np.int32)


def plot_timeline(video, gt, pred, label_to_id, id_to_label, out_path, width=1600):
    L = min(len(gt), len(pred))
    gt = gt[:L]
    pred = pred[:L]

    if L <= 0:
        return

    gt_ids = to_ids(gt, label_to_id, id_to_label)
    pr_ids = to_ids(pred, label_to_id, id_to_label)

    W = min(width, L)
    idx = np.linspace(0, L - 1, W).astype(int)
    mat = np.vstack([gt_ids[idx], pr_ids[idx]])

    used = sorted(set(mat.reshape(-1).tolist()))
    vmax = max(max(used), 19)

    fig_h = 3.0 + 0.15 * min(len(used), 20)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="tab20", vmin=0, vmax=vmax)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GT", "Pred"])
    ax.set_xlabel(f"compressed time axis, original T={L}")
    ax.set_title(video)

    ax.set_xticks(np.linspace(0, W - 1, 6))
    ax.set_xticklabels([str(int(x)) for x in np.linspace(0, L - 1, 6)])

    cmap = plt.get_cmap("tab20")
    handles = []
    for cid in used[:24]:
        lab = id_to_label.get(cid, str(cid))
        handles.append(Patch(color=cmap(cid % 20), label=f"{cid}:{lab}"))

    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ROOT", "/path/to/MS-TCN2"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--a4_dir", default="/path/to/MS-TCN2/data/sics155_100_a4_geo_assets_sam2_yolo_tiptrack_cons_compat/npz")
    ap.add_argument("--maskshape_dir", default="/path/to/MS-TCN2/data/sics155_100_a5_maskshape_assets_v1_safe/npz")
    args = ap.parse_args()

    root = Path(args.root)
    data_dir = root / "data" / args.dataset
    gt_dir = data_dir / "groundTruth"
    pred_dir = root / "results" / f"tecno_{args.dataset}" / f"split_{args.split}"
    split_file = data_dir / "splits" / f"test.split{args.split}.bundle"

    a4_dir = Path(args.a4_dir)
    maskshape_dir = Path(args.maskshape_dir)

    out_dir = root / "diagnostics" / f"{args.dataset}_split{args.split}_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_label, label_to_id = parse_mapping(data_dir / "mapping.txt")

    videos = [x.strip() for x in split_file.read_text().splitlines() if x.strip()]

    print("DATASET:", args.dataset)
    print("SPLIT:", args.split)
    print("test videos:", len(videos))
    print("GT_DIR:", gt_dir)
    print("PRED_DIR:", pred_dir)
    print("OUT_DIR:", out_dir)

    rows = []
    gt_counter = Counter()
    pred_counter = Counter()

    total_correct = 0
    total_frames = 0

    edit_values = []

    f_counts = {
        0.10: [0, 0, 0],
        0.25: [0, 0, 0],
        0.50: [0, 0, 0],
    }

    for video in videos:
        gp = find_gt(gt_dir, video)
        pp = find_pred(pred_dir, video)

        if gp is None:
            print("[MISS GT]", video)
            continue
        if pp is None:
            print("[MISS PRED]", video)
            continue

        gt = read_seq(gp)
        pr = read_seq(pp)

        L = min(len(gt), len(pr))
        if len(gt) != len(pr):
            print(f"[LEN MISMATCH] {video}: gt={len(gt)} pred={len(pr)} use={L}")

        gt = gt[:L]
        pr = pr[:L]

        gt_counter.update(gt)
        pred_counter.update(pr)

        gs = segments(gt)
        ps = segments(pr)

        correct = sum(a == b for a, b in zip(gt, pr))
        total_correct += correct
        total_frames += L

        acc = 100.0 * correct / max(L, 1)
        ed = edit_score(gt, pr)
        edit_values.append(ed)

        f10_counts = segment_f1_counts(gt, pr, 0.10)
        f25_counts = segment_f1_counts(gt, pr, 0.25)
        f50_counts = segment_f1_counts(gt, pr, 0.50)

        for ov, c in [(0.10, f10_counts), (0.25, f25_counts), (0.50, f50_counts)]:
            f_counts[ov][0] += c[0]
            f_counts[ov][1] += c[1]
            f_counts[ov][2] += c[2]

        f10 = f1_from_counts(*f10_counts)
        f25 = f1_from_counts(*f25_counts)
        f50 = f1_from_counts(*f50_counts)

        a4_p = a4_dir / f"{Path(video).stem}.npz"
        mask_p = maskshape_dir / f"{Path(video).stem}.npz"

        row = {
            "video": Path(video).stem,
            "T": L,
            "acc": acc,
            "edit": ed,
            "f1@10": f10,
            "f1@25": f25,
            "f1@50": f50,
            "gt_seg": len(gs),
            "pred_seg": len(ps),
            "seg_ratio": len(ps) / max(len(gs), 1),
            "gt_med_len": float(np.median([e - s for _, s, e in gs])) if gs else 0.0,
            "pred_med_len": float(np.median([e - s for _, s, e in ps])) if ps else 0.0,
            "eye_valid": safe_mean_npz(a4_p, "eye_valid"),
            "inst_valid_any": safe_mean_npz(a4_p, "inst_valid_any"),
            "eye_radius": safe_mean_npz(a4_p, "eye_radius"),
            "mask_shape_valid": safe_mean_npz(mask_p, "mask_shape_valid"),
        }

        rows.append(row)

    if not rows:
        print("[ERROR] no valid rows")
        return

    eval_like = {
        "Acc": 100.0 * total_correct / max(total_frames, 1),
        "Edit": float(np.mean(edit_values)),
        "F1@0.10": f1_from_counts(*f_counts[0.10]),
        "F1@0.25": f1_from_counts(*f_counts[0.25]),
        "F1@0.50": f1_from_counts(*f_counts[0.50]),
    }

    print("\n===== Eval-like metrics from this parser =====")
    for k, v in eval_like.items():
        print(f"{k}: {v:.4f}")

    print("\n===== Per-video mean diagnostics =====")
    for k in [
        "T", "acc", "edit", "f1@10", "f1@25", "f1@50",
        "gt_seg", "pred_seg", "seg_ratio",
        "gt_med_len", "pred_med_len",
        "eye_valid", "inst_valid_any", "mask_shape_valid",
    ]:
        vals = np.asarray([r[k] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        print(f"{k:18s}: {vals.mean():.4f} ± {vals.std(ddof=0):.4f}")

    print("\n===== Correlation with F1@25 / Edit =====")
    for q in ["eye_valid", "inst_valid_any", "mask_shape_valid", "eye_radius", "seg_ratio", "pred_seg", "pred_med_len"]:
        print(
            f"{q:18s} "
            f"corr_f25={corr([r[q] for r in rows], [r['f1@25'] for r in rows]): .4f} "
            f"corr_edit={corr([r[q] for r in rows], [r['edit'] for r in rows]): .4f}"
        )

    print("\n===== Worst F1@25 videos =====")
    for r in sorted(rows, key=lambda x: x["f1@25"])[:12]:
        print(
            f"{r['video']:12s} T={r['T']:6d} acc={r['acc']:6.2f} edit={r['edit']:6.2f} "
            f"f25={r['f1@25']:6.2f} f50={r['f1@50']:6.2f} "
            f"gt_seg={r['gt_seg']:3d} pred_seg={r['pred_seg']:4d} ratio={r['seg_ratio']:.2f} "
            f"eye={r['eye_valid']:.3f} inst={r['inst_valid_any']:.3f} maskv={r['mask_shape_valid']:.3f}"
        )

    print("\n===== Most fragmented videos =====")
    for r in sorted(rows, key=lambda x: x["seg_ratio"], reverse=True)[:12]:
        print(
            f"{r['video']:12s} T={r['T']:6d} acc={r['acc']:6.2f} edit={r['edit']:6.2f} "
            f"f25={r['f1@25']:6.2f} gt_seg={r['gt_seg']:3d} "
            f"pred_seg={r['pred_seg']:4d} ratio={r['seg_ratio']:.2f} "
            f"pred_med={r['pred_med_len']:.1f}"
        )

    print("\n===== Classes missing in prediction =====")
    missing = sorted(set(gt_counter) - set(pred_counter), key=lambda x: label_to_id.get(x, 999))
    if not missing:
        print("[OK] no missing GT class")
    else:
        for k in missing:
            print(f"{k:30s} gt_count={gt_counter[k]}")

    print("\n===== Class distribution GT vs Pred =====")
    all_cls = sorted(set(gt_counter) | set(pred_counter), key=lambda x: label_to_id.get(x, 999))
    for c in all_cls:
        print(f"{c:30s} gt={gt_counter[c]:8d} pred={pred_counter[c]:8d} diff={pred_counter[c]-gt_counter[c]:8d}")

    # save CSV
    csv_path = out_dir / "per_video_diagnosis_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    dist_path = out_dir / "class_distribution_v2.csv"
    with dist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "gt_count", "pred_count", "pred_minus_gt"])
        for c in all_cls:
            w.writerow([c, gt_counter[c], pred_counter[c], pred_counter[c] - gt_counter[c]])

    # plots
    rows_rank = sorted(rows, key=lambda x: x["seg_ratio"], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.arange(len(rows_rank))
    ax.bar(xs, [r["seg_ratio"] for r in rows_rank])
    ax.axhline(1.0, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["video"] for r in rows_rank], rotation=90, fontsize=7)
    ax.set_ylabel("Pred segments / GT segments")
    ax.set_title("Fragmentation ratio by test video")
    fig.tight_layout()
    fig.savefig(out_dir / "segment_ratio_rank_v2.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, q in zip(axes, ["eye_valid", "inst_valid_any", "mask_shape_valid"]):
        ax.scatter([r[q] for r in rows], [r["f1@25"] for r in rows], s=24)
        ax.set_xlabel(q)
        ax.set_ylabel("F1@25")
        ax.set_title(f"{q} vs F1@25")
    fig.tight_layout()
    fig.savefig(out_dir / "asset_quality_vs_f1@25_v2.png", dpi=160)
    plt.close(fig)

    # timelines
    worst = sorted(rows, key=lambda x: x["f1@25"])[:8]
    frag = sorted(rows, key=lambda x: x["seg_ratio"], reverse=True)[:8]

    for rank, r in enumerate(worst, 1):
        video = r["video"]
        gp = find_gt(gt_dir, video)
        pp = find_pred(pred_dir, video)
        if gp and pp:
            plot_timeline(
                f"worstF25_{rank}_{video} | acc={r['acc']:.1f} edit={r['edit']:.1f} f25={r['f1@25']:.1f}",
                read_seq(gp), read_seq(pp),
                label_to_id, id_to_label,
                out_dir / f"timeline_worstF25_{rank}_{video}.png"
            )

    for rank, r in enumerate(frag, 1):
        video = r["video"]
        gp = find_gt(gt_dir, video)
        pp = find_pred(pred_dir, video)
        if gp and pp:
            plot_timeline(
                f"fragmented_{rank}_{video} | ratio={r['seg_ratio']:.2f} f25={r['f1@25']:.1f}",
                read_seq(gp), read_seq(pp),
                label_to_id, id_to_label,
                out_dir / f"timeline_fragmented_{rank}_{video}.png"
            )

    print("\n[SAVED]")
    print("out_dir:", out_dir)
    print("per-video csv:", csv_path)
    print("class dist:", dist_path)


if __name__ == "__main__":
    main()

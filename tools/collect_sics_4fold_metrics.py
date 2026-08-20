import os
import argparse
import re
import subprocess
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ROOT", "/path/to/MS-TCN2"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", default="1 2 3 4")
    args = ap.parse_args()

    root = Path(args.root)
    splits = [int(x) for x in args.splits.split()]
    rows = []

    for sp in splits:
        r = subprocess.run(
            ["python", str(root / "eval.py"), "--dataset", args.dataset, "--split", str(sp)],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        vals = {"split": sp}
        for k in ["Acc", "Edit", "F1@0.10", "F1@0.25", "F1@0.50"]:
            m = re.search(rf"{re.escape(k)}:\s*([0-9.]+)", r.stdout)
            if m:
                vals[k] = float(m.group(1))

        if len(vals) < 6:
            print(f"[WARN] incomplete split {sp}")
            print(r.stdout[-2000:])
            continue

        rows.append(vals)

    print("DATASET:", args.dataset)
    print("split,Acc,Edit,F1@10,F1@25,F1@50")
    for x in rows:
        print(
            f"{x['split']},"
            f"{x['Acc']:.4f},"
            f"{x['Edit']:.4f},"
            f"{x['F1@0.10']:.4f},"
            f"{x['F1@0.25']:.4f},"
            f"{x['F1@0.50']:.4f}"
        )

    print()
    print("===== Mean ± Std =====")
    for k in ["Acc", "Edit", "F1@0.10", "F1@0.25", "F1@0.50"]:
        arr = np.array([x[k] for x in rows], dtype=float)
        print(f"{k:8s}: {arr.mean():.4f} ± {arr.std(ddof=0):.4f}")

if __name__ == "__main__":
    main()

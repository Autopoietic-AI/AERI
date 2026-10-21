#!/usr/bin/env bash
set -euo pipefail

REL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:?Usage: run_4way_ablation_seed42.sh <dataset_name> [epochs]}"
EPOCHS="${2:-100}"
ROOT="$REL/runtime_root"
BASE="$ROOT/data/$NAME"
MAIN="$REL/code/stagefusion/ablation/main_cat101_a5_tig_stagefusion_ablation.py"
EVAL="$REL/code/evaluation/eval.py"
GENEVAL="$REL/launchers/evaluate_predictions.py"

python "$REL/launchers/validate_dataset_contract.py" "$BASE"

export PYTHONPATH="$REL/code:$REL/code/stagefusion/ablation:${PYTHONPATH:-}"
export A5_GLOBAL_SEED=42
export A5_TIG_DILATED_OFFSETS="-16,-8,-4,-2,0,2,4,8,16"

mapfile -t SPLITS < <(
  find "$BASE/splits" -maxdepth 1 -type f -name 'test.split*.bundle' -printf '%f\n' |
  sed -E 's/test\.split([0-9]+)\.bundle/\1/' | sort -n
)
[[ "${#SPLITS[@]}" -gt 0 ]] || { echo "[ERROR] no test split bundles"; exit 2; }

run_variant() {
  local variant="$1"
  local alias="${NAME}_release_seed42_${variant}"
  local data_alias="$ROOT/data/$alias"
  rm -rf "$data_alias"
  mkdir -p "$data_alias"
  ln -s "$(readlink -f "$BASE/mapping.txt")" "$data_alias/mapping.txt"
  ln -s "$(readlink -f "$BASE/groundTruth")" "$data_alias/groundTruth"
  ln -s "$(readlink -f "$BASE/splits")" "$data_alias/splits"

  unset A5_VISUAL_ONLY A5_ABLATE_TIG
  case "$variant" in
    visual_only)
      export A5_VISUAL_ONLY=1
      export A5_STAGE_GATE_RESIDUAL=0.25
      ;;
    local_structured)
      export A5_ABLATE_TIG=1
      export A5_STAGE_GATE_RESIDUAL=0
      ;;
    wtig_fixed)
      export A5_STAGE_GATE_RESIDUAL=0
      ;;
    full)
      export A5_STAGE_GATE_RESIDUAL=0.25
      ;;
    *) echo "bad variant: $variant"; exit 3;;
  esac

  for split in "${SPLITS[@]}"; do
    echo "================================================================"
    echo "[RUN] $variant split=$split seed=42"
    echo "================================================================"
    model_dir="$ROOT/models/tecno_${alias}/split_${split}"
    if [[ ! -f "$model_dir/epoch-${EPOCHS}.model" ]]; then
      python -u "$MAIN" \
        --root "$ROOT" \
        --dataset "$alias" \
        --m8_feature_dir "$BASE/features" \
        --m7_asset_dir "$BASE/geo_assets" \
        --maskshape_dir "$BASE/maskshape" \
        --split "$split" \
        --action train \
        --num_epochs "$EPOCHS" \
        --learning_rate 5e-4 \
        --num_stages 4 \
        --num_layers 10 \
        --num_f_maps 64 \
        --main_dim 256 \
        --guide_dim 64 \
        --beta_max 0.20 \
        --gate_l1 0.001 \
        --smooth_w 0.15 \
        --expected_dim 384 \
        2>&1 | tee "$ROOT/logs/${alias}_train_split${split}.log"
    else
      echo "[SKIP TRAIN] endpoint exists: $model_dir/epoch-${EPOCHS}.model"
    fi

    python -u "$MAIN" \
      --root "$ROOT" \
      --dataset "$alias" \
      --m8_feature_dir "$BASE/features" \
      --m7_asset_dir "$BASE/geo_assets" \
      --maskshape_dir "$BASE/maskshape" \
      --split "$split" \
      --action predict \
      --num_epochs "$EPOCHS" \
      --learning_rate 5e-4 \
      --num_stages 4 \
      --num_layers 10 \
      --num_f_maps 64 \
      --main_dim 256 \
      --guide_dim 64 \
      --beta_max 0.20 \
      --gate_l1 0.001 \
      --smooth_w 0.15 \
      --expected_dim 384 \
      2>&1 | tee "$ROOT/logs/${alias}_predict_split${split}.log"

    python "$GENEVAL" \
      --eval_py "$EVAL" \
      --gt_dir "$BASE/groundTruth" \
      --bundle "$BASE/splits/test.split${split}.bundle" \
      --pred_dir "$ROOT/results/tecno_${alias}/split_${split}" \
      --out_csv "$ROOT/results/tecno_${alias}/split_${split}_metrics.csv"

    # Keep only the two formal endpoints after successful prediction/evaluation.
    if [[ -f "$model_dir/best.model" && -f "$model_dir/epoch-${EPOCHS}.model" ]]; then
      find "$model_dir" -maxdepth 1 -type f -name 'epoch-*.model' \
        ! -name "epoch-${EPOCHS}.model" -delete
    fi
  done
}

for v in visual_only local_structured wtig_fixed full; do
  run_variant "$v"
done

python - "$ROOT" "$NAME" "${SPLITS[@]}" <<'PY'
import sys, csv, numpy as np
from pathlib import Path
root=Path(sys.argv[1]); name=sys.argv[2]; splits=[int(x) for x in sys.argv[3:]]
variants=["visual_only","local_structured","wtig_fixed","full"]
metrics=["Acc","Edit","F1@0.10","F1@0.25","F1@0.50"]
outdir=root/"results"/f"{name}_release_seed42_ablation_summary"
outdir.mkdir(parents=True,exist_ok=True)
rows=[]
for v in variants:
    vals={m:[] for m in metrics}
    for s in splits:
        p=root/"results"/f"tecno_{name}_release_seed42_{v}"/f"split_{s}_metrics.csv"
        with p.open() as f:
            r=next(csv.DictReader(f))
        row={"variant":v,"split":s}
        for m in metrics:
            x=float(r[m]); vals[m].append(x); row[m]=x
        rows.append(row)
with (outdir/"per_fold.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["variant","split"]+metrics); w.writeheader(); w.writerows(rows)
with (outdir/"summary.csv").open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["variant"]+[x for m in metrics for x in (m+"_mean",m+"_std")])
    for v in variants:
        r=[v]
        for m in metrics:
            xs=[x[m] for x in rows if x["variant"]==v]
            r += [float(np.mean(xs)), float(np.std(xs))]
        w.writerow(r)
print("[DONE]",outdir/"summary.csv")
PY

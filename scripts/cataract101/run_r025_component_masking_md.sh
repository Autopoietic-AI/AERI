#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/path/to/MS-TCN2}
MAIN_PY=${ROOT}/m10/a5_tig_stagefusion/main_cat101_a5_tig_stagefusion_ablation.py
MODEL_PY=${ROOT}/m10/a5_tig_stagefusion/model_a5_tig_stagefusion_ablation.py
MASKSHAPE_DIR=${ROOT}/data/cataract101_m10_a5_maskshape_assets_v1_safe/npz

LOG_DIR=${ROOT}/logs/m10_r025_component_masking
mkdir -p "${LOG_DIR}"

CSV_FILE=${LOG_DIR}/r025_component_masking_results.csv
MD_FILE=${LOG_DIR}/r025_component_masking_results.md

FORCE=${FORCE:-0}
SPLITS=${SPLITS:-"1 2 3 4"}

NUM_EPOCHS=100
LR=0.0005
NUM_F_MAPS=64
SMOOTH_W=0.15
EXPECTED_DIM=384

# Common official StageFusion-r0.25 setting
export A5_DFUSION_F_DIM=48
export A5_DFUSION_E_LOCAL_DIM=12
export A5_TIG_E_TEMP_DIM=12
export A5_DFUSION_A_DIM=4
export A5_DFUSION_HIDDEN=128

export A5_TIG_Q_DIM=64
export A5_TIG_WINDOW=9
export A5_TIG_DILATED_OFFSETS="-16,-8,-4,-2,0,2,4,8,16"

export A5_DFUSION_FIXED_GATE=1
export A5_MEMORY_SIGMA=0.22

if [ "${FORCE}" = "1" ]; then
  rm -f "${CSV_FILE}" "${MD_FILE}"
fi

if [ ! -f "${CSV_FILE}" ]; then
  echo "variant,split,acc,edit,f1_10,f1_25,f1_50,eval_log" > "${CSV_FILE}"
fi

reset_ablation_env () {
  unset A5_VISUAL_ONLY || true
  unset A5_ABLATE_TIG || true
  unset A5_ABLATE_LOCAL_INTERACTION || true
  unset A5_ABLATE_INTERACTION || true
  unset A5_ABLATE_AUX_BRANCH || true
  export A5_STAGE_GATE_RESIDUAL=0.25
}

set_variant_env () {
  VARIANT=$1
  reset_ablation_env

  case "${VARIANT}" in
    full_r025)
      export A5_STAGE_GATE_RESIDUAL=0.25
      ;;
    r0_fixed_gate)
      export A5_STAGE_GATE_RESIDUAL=0
      ;;
    wo_tig)
      export A5_ABLATE_TIG=1
      export A5_STAGE_GATE_RESIDUAL=0.25
      ;;
    wo_local_interaction)
      export A5_ABLATE_LOCAL_INTERACTION=1
      export A5_STAGE_GATE_RESIDUAL=0.25
      ;;
    wo_interaction_branch)
      export A5_ABLATE_INTERACTION=1
      export A5_STAGE_GATE_RESIDUAL=0.25
      ;;
    wo_aux_branch)
      export A5_ABLATE_AUX_BRANCH=1
      export A5_STAGE_GATE_RESIDUAL=0.25
      ;;
    visual_only)
      export A5_VISUAL_ONLY=1
      export A5_STAGE_GATE_RESIDUAL=0.25
      ;;
    *)
      echo "[ERROR] Unknown variant: ${VARIANT}"
      exit 1
      ;;
  esac
}

extract_metric () {
  METRIC=$1
  LOG=$2
  grep -E "^${METRIC}:" "${LOG}" | tail -1 | awk -F':' '{gsub(/[[:space:]]/, "", $2); print $2}'
}

update_csv_result () {
  VARIANT=$1
  SPLIT=$2
  EVAL_LOG=$3

  ACC=$(extract_metric "Acc" "${EVAL_LOG}")
  EDIT=$(extract_metric "Edit" "${EVAL_LOG}")
  F10=$(extract_metric "F1@0.10" "${EVAL_LOG}")
  F25=$(extract_metric "F1@0.25" "${EVAL_LOG}")
  F50=$(extract_metric "F1@0.50" "${EVAL_LOG}")

  python - "${CSV_FILE}" "${VARIANT}" "${SPLIT}" "${ACC}" "${EDIT}" "${F10}" "${F25}" "${F50}" "${EVAL_LOG}" <<'PY'
import sys
import pandas as pd
from pathlib import Path

csv_path = Path(sys.argv[1])
variant, split = sys.argv[2], int(sys.argv[3])
acc, edit, f10, f25, f50 = map(float, sys.argv[4:9])
eval_log = sys.argv[9]

if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame(columns=["variant","split","acc","edit","f1_10","f1_25","f1_50","eval_log"])

if len(df):
    df = df[~((df["variant"] == variant) & (df["split"] == split))]

new = pd.DataFrame([{
    "variant": variant,
    "split": split,
    "acc": acc,
    "edit": edit,
    "f1_10": f10,
    "f1_25": f25,
    "f1_50": f50,
    "eval_log": eval_log
}])

df = pd.concat([df, new], ignore_index=True)
df = df.sort_values(["split", "variant"])
df.to_csv(csv_path, index=False)
PY
}

write_markdown () {
  python - "${CSV_FILE}" "${MD_FILE}" <<'PY'
import sys
import pandas as pd
from pathlib import Path

csv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])

df = pd.read_csv(csv_path)
df = df.drop_duplicates(subset=["variant", "split"], keep="last")

metrics = ["acc", "edit", "f1_10", "f1_25", "f1_50"]
for m in metrics:
    df[m] = pd.to_numeric(df[m], errors="coerce")

variant_order = [
    "full_r025",
    "r0_fixed_gate",
    "wo_tig",
    "wo_local_interaction",
    "wo_interaction_branch",
    "wo_aux_branch",
    "visual_only",
]
df["variant"] = pd.Categorical(df["variant"], categories=variant_order, ordered=True)
df = df.sort_values(["variant", "split"])

summary_rows = []
for variant, g in df.groupby("variant", sort=False, observed=True):
    row = {"variant": variant, "n": len(g)}
    for m in metrics:
        row[m] = f"{g[m].mean():.4f} ± {g[m].std(ddof=0):.4f}"
    summary_rows.append(row)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# GeoIG-StageFusion-r0.25 Component Masking Results\n\n")
    f.write("This is a checkpoint-controlled inference-time component masking analysis.\n\n")
    f.write("Official checkpoints are reused from:\n\n")
    f.write("`models/tecno_cataract101_m10_a5_tig_stagefusion_r025_s{split}_best/split_{split}/epoch-100.model`\n\n")

    f.write("## Experiment Setting\n\n")
    f.write("| Item | Value |\n")
    f.write("|---|---|\n")
    f.write("| Dataset | Cataract101 |\n")
    f.write("| Evaluation | 4-fold leave-one-split-out |\n")
    f.write("| Training | No retraining; official StageFusion-r0.25 checkpoints |\n")
    f.write("| Epoch checkpoint | epoch-100.model |\n")
    f.write("| TIG offsets | -16,-8,-4,-2,0,2,4,8,16 |\n")
    f.write("| Full residual | r=0.25 |\n\n")

    f.write("## Variant Definition\n\n")
    f.write("| Variant | Meaning |\n")
    f.write("|---|---|\n")
    f.write("| full_r025 | Full GeoIG-StageFusion-r0.25 |\n")
    f.write("| r0_fixed_gate | Set residual strength to r=0, using base gate only |\n")
    f.write("| wo_tig | Mask temporal interaction token from TIG |\n")
    f.write("| wo_local_interaction | Mask local eye-anchored structured interaction token |\n")
    f.write("| wo_interaction_branch | Mask the entire interaction branch |\n")
    f.write("| wo_aux_branch | Mask auxiliary/reliability branch |\n")
    f.write("| visual_only | Use visual branch only; force gate=[1,0,0] |\n\n")

    f.write("## Raw Split Results\n\n")
    f.write("| Variant | Split | Acc | Edit | F1@0.10 | F1@0.25 | F1@0.50 | Log |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for _, r in df.iterrows():
        f.write(
            f"| {r['variant']} | {int(r['split'])} | "
            f"{r['acc']:.4f} | {r['edit']:.4f} | {r['f1_10']:.4f} | "
            f"{r['f1_25']:.4f} | {r['f1_50']:.4f} | `{r['eval_log']}` |\n"
        )

    f.write("\n\n## Mean ± Std Summary\n\n")
    f.write("| Variant | N | Acc | Edit | F1@0.10 | F1@0.25 | F1@0.50 |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in summary_rows:
        f.write(
            f"| {r['variant']} | {r['n']} | {r['acc']} | {r['edit']} | "
            f"{r['f1_10']} | {r['f1_25']} | {r['f1_50']} |\n"
        )

print(f"[OK] Markdown written: {md_path}")
PY
}

run_one () {
  VARIANT=$1
  SPLIT=$2

  set_variant_env "${VARIANT}"

  SRC="cataract101_m10_a5_tig_stagefusion_r025_s${SPLIT}_best"
  ALIAS="cataract101_m10_r025_mask_${VARIANT}_s${SPLIT}"

  PRED_LOG=${LOG_DIR}/predict_${ALIAS}_split${SPLIT}.log
  EVAL_LOG=${LOG_DIR}/eval_${ALIAS}_split${SPLIT}.log

  echo
  echo "============================================================"
  echo "VARIANT=${VARIANT}"
  echo "SPLIT=${SPLIT}"
  echo "SRC=${SRC}"
  echo "ALIAS=${ALIAS}"
  echo "A5_STAGE_GATE_RESIDUAL=${A5_STAGE_GATE_RESIDUAL:-NA}"
  echo "A5_VISUAL_ONLY=${A5_VISUAL_ONLY:-0}"
  echo "A5_ABLATE_TIG=${A5_ABLATE_TIG:-0}"
  echo "A5_ABLATE_LOCAL_INTERACTION=${A5_ABLATE_LOCAL_INTERACTION:-0}"
  echo "A5_ABLATE_INTERACTION=${A5_ABLATE_INTERACTION:-0}"
  echo "A5_ABLATE_AUX_BRANCH=${A5_ABLATE_AUX_BRANCH:-0}"
  echo "============================================================"

  if [ -f "${EVAL_LOG}" ] && [ "${FORCE}" != "1" ]; then
    echo "[SKIP] Existing eval log found: ${EVAL_LOG}"
    update_csv_result "${VARIANT}" "${SPLIT}" "${EVAL_LOG}"
    write_markdown
    return
  fi

  # Link data alias
  ln -sfn "${ROOT}/data/${SRC}" "${ROOT}/data/${ALIAS}"

  # Link official checkpoint directory
  mkdir -p "${ROOT}/models/tecno_${ALIAS}"
  ln -sfn "${ROOT}/models/tecno_${SRC}/split_${SPLIT}" \
          "${ROOT}/models/tecno_${ALIAS}/split_${SPLIT}"

  # Predict
  python "${MAIN_PY}" \
    --action predict \
    --dataset "${ALIAS}" \
    --split "${SPLIT}" \
    --num_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LR}" \
    --num_f_maps "${NUM_F_MAPS}" \
    --smooth_w "${SMOOTH_W}" \
    --expected_dim "${EXPECTED_DIM}" \
    --m8_feature_dir "${ROOT}/data/${ALIAS}/features" \
    --m7_asset_dir "${ROOT}/data/${ALIAS}/geo_assets" \
    --maskshape_dir "${MASKSHAPE_DIR}" \
    2>&1 | tee "${PRED_LOG}"

  # Prepare eval link
  EVAL_PARENT="${ROOT}/results/${ALIAS}"
  EVAL_DIR="${EVAL_PARENT}/split_${SPLIT}"
  mkdir -p "${EVAL_PARENT}"
  rm -rf "${EVAL_DIR}"

  REAL_RESULT_DIR="$(find "${ROOT}/results" -maxdepth 3 -type d -path "*/split_${SPLIT}" | grep "tecno_${ALIAS}" | sort | tail -1 || true)"
  if [ -z "${REAL_RESULT_DIR}" ]; then
    echo "[ERROR] Cannot find prediction dir for ${ALIAS}"
    exit 1
  fi

  ln -s "$(realpath "${REAL_RESULT_DIR}")" "${EVAL_DIR}"

  # Eval
  python "${ROOT}/eval.py" \
    --dataset "${ALIAS}" \
    --split "${SPLIT}" \
    2>&1 | tee "${EVAL_LOG}"

  echo
  echo "===== FINAL RESULT ${VARIANT} split ${SPLIT} ====="
  grep -E "Acc:|Edit:|F1@0.10:|F1@0.25:|F1@0.50:" "${EVAL_LOG}"

  update_csv_result "${VARIANT}" "${SPLIT}" "${EVAL_LOG}"
  write_markdown
}

echo "===== Compile check ====="
python -m py_compile "${MODEL_PY}"
python -m py_compile "${MAIN_PY}"

VARIANTS=(
  "full_r025"
  "r0_fixed_gate"
  "wo_tig"
  "wo_local_interaction"
  "wo_interaction_branch"
  "wo_aux_branch"
  "visual_only"
)

for SPLIT in ${SPLITS}
do
  for VARIANT in "${VARIANTS[@]}"
  do
    run_one "${VARIANT}" "${SPLIT}"
  done
done

echo
echo "===== ALL COMPONENT MASKING DONE ====="
echo "CSV: ${CSV_FILE}"
echo "MD : ${MD_FILE}"
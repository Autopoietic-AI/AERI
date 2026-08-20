#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/path/to/MS-TCN2}
MAIN_PY=${ROOT}/m10/a5_tig_stagefusion/main_cat101_a5_tig_stagefusion.py
MODEL_PY=${ROOT}/m10/a5_tig_stagefusion/model_a5_tig_stagefusion.py
MASKSHAPE_DIR=${ROOT}/data/cataract101_m10_a5_maskshape_assets_v1_safe/npz
LOG_DIR=${ROOT}/logs/m10

mkdir -p "${LOG_DIR}"

# =========================
# Model hyperparameters
# =========================
export A5_DFUSION_F_DIM=48
export A5_DFUSION_E_LOCAL_DIM=12
export A5_TIG_E_TEMP_DIM=12
export A5_DFUSION_A_DIM=4
export A5_DFUSION_HIDDEN=128

export A5_TIG_Q_DIM=64
export A5_TIG_WINDOW=9
export A5_TIG_DILATED_OFFSETS="-16,-8,-4,-2,0,2,4,8,16"

# StageFusion-r0.25
export A5_DFUSION_FIXED_GATE=1
export A5_STAGE_GATE_RESIDUAL=0.25
export A5_MEMORY_SIGMA=0.22

echo "============================================================"
echo "[A5-TIG-Dfusion-v7b StageFusion-r0.25 Multi-split]"
echo "MAIN_PY=${MAIN_PY}"
echo "MODEL_PY=${MODEL_PY}"
echo "MASKSHAPE_DIR=${MASKSHAPE_DIR}"
echo "A5_STAGE_GATE_RESIDUAL=${A5_STAGE_GATE_RESIDUAL}"
echo "A5_MEMORY_SIGMA=${A5_MEMORY_SIGMA}"
echo "============================================================"

python -m py_compile "${MODEL_PY}"
python -m py_compile "${MAIN_PY}"

for SPLIT in 2 3 4
do
  DATASET="cataract101_m10_a5_tig_stagefusion_r025_s${SPLIT}_best"

  echo
  echo "################################################################"
  echo "### RUN StageFusion-r0.25 split ${SPLIT}"
  echo "### DATASET=${DATASET}"
  echo "################################################################"

  echo
  echo "===== Check dataset ====="
  ls -lh "${ROOT}/data/${DATASET}"
  ls -lh "${ROOT}/data/${DATASET}/splits/train.split${SPLIT}.bundle"
  ls -lh "${ROOT}/data/${DATASET}/splits/test.split${SPLIT}.bundle"

  echo
  echo "===== Train split ${SPLIT} ====="
  python "${MAIN_PY}" \
    --action train \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --num_epochs 100 \
    --learning_rate 0.0005 \
    --num_f_maps 64 \
    --smooth_w 0.15 \
    --expected_dim 384 \
    --m8_feature_dir "${ROOT}/data/${DATASET}/features" \
    --m7_asset_dir "${ROOT}/data/${DATASET}/geo_assets" \
    --maskshape_dir "${MASKSHAPE_DIR}" \
    2>&1 | tee "${LOG_DIR}/train_${DATASET}_split${SPLIT}.log"

  echo
  echo "===== Predict split ${SPLIT} ====="
  python "${MAIN_PY}" \
    --action predict \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --num_epochs 100 \
    --learning_rate 0.0005 \
    --num_f_maps 64 \
    --smooth_w 0.15 \
    --expected_dim 384 \
    --m8_feature_dir "${ROOT}/data/${DATASET}/features" \
    --m7_asset_dir "${ROOT}/data/${DATASET}/geo_assets" \
    --maskshape_dir "${MASKSHAPE_DIR}" \
    2>&1 | tee "${LOG_DIR}/predict_${DATASET}_split${SPLIT}.log"

  echo
  echo "===== Eval split ${SPLIT} ====="

  EVAL_PARENT="${ROOT}/results/${DATASET}"
  EVAL_DIR="${EVAL_PARENT}/split_${SPLIT}"

  mkdir -p "${EVAL_PARENT}"
  rm -rf "${EVAL_DIR}"

  REAL_RESULT_DIR="$(find "${ROOT}/results" -maxdepth 3 -type d -path "*/split_${SPLIT}" | grep "tecno_${DATASET}" | sort | tail -1 || true)"

  if [ -z "${REAL_RESULT_DIR}" ]; then
    echo "[ERROR] Cannot find prediction dir for ${DATASET}"
    echo "[DEBUG] Available result dirs:"
    find "${ROOT}/results" -maxdepth 3 -type d | grep "${DATASET}" || true
    exit 1
  fi

  echo "[INFO] REAL_RESULT_DIR=${REAL_RESULT_DIR}"
  ln -s "$(realpath "${REAL_RESULT_DIR}")" "${EVAL_DIR}"

  python "${ROOT}/eval.py" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    2>&1 | tee "${LOG_DIR}/eval_${DATASET}_split${SPLIT}.log"

  echo
  echo "===== FINAL RESULT StageFusion-r0.25 split ${SPLIT} ====="
  grep -E "Acc:|Edit:|F1@0.10:|F1@0.25:|F1@0.50:" "${LOG_DIR}/eval_${DATASET}_split${SPLIT}.log"
done

echo
echo "===== ALL DONE: StageFusion-r0.25 split 2/3/4 ====="

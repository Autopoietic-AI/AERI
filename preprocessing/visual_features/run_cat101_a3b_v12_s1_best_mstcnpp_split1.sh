#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/path/to/MS-TCN2}
DATASET=cataract101_m10_a3b_v12_dualgate_s1_best
SPLIT=1
EPOCHS=100
LR=0.0005
LOG_DIR=${ROOT}/logs/m10
MAIN_PY=${ROOT}/m8/main_m8_v12_mstcnpp.py

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "[RUN] M10-A3b-v12-dualgate-s1-best + MS-TCN++"
echo "DATASET=${DATASET}"
echo "SPLIT=${SPLIT}"
echo "EPOCHS=${EPOCHS}"
echo "MAIN=${MAIN_PY}"
echo "============================================================"

echo "===== CHECK DATASET ====="
ls "data/${DATASET}/mapping.txt"
find "data/${DATASET}/features" -maxdepth 1 -name "case_*.npy" | wc -l
wc -l "data/${DATASET}/splits/train.split${SPLIT}.bundle"
wc -l "data/${DATASET}/splits/test.split${SPLIT}.bundle"

echo "===== TRAIN ====="
python -u "${MAIN_PY}" \
  --action train \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --num_epochs "${EPOCHS}" \
  --learning_rate "${LR}" \
  --num_layers_PG 11 \
  --num_layers_R 10 \
  --num_R 4 \
  --num_f_maps 64 \
  --smooth_w 0.15 \
  --expected_dim 384 \
  2>&1 | tee "${LOG_DIR}/train_${DATASET}_split${SPLIT}.log"

echo "===== PREDICT ====="
python -u "${MAIN_PY}" \
  --action predict \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --num_epochs "${EPOCHS}" \
  --learning_rate "${LR}" \
  --num_layers_PG 11 \
  --num_layers_R 10 \
  --num_R 4 \
  --num_f_maps 64 \
  --smooth_w 0.15 \
  --expected_dim 384 \
  2>&1 | tee "${LOG_DIR}/predict_${DATASET}_split${SPLIT}.log"

echo "===== EVAL ====="
python "${ROOT}/eval.py" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  2>&1 | tee "${LOG_DIR}/eval_${DATASET}_split${SPLIT}.log"

echo "===== RESULT ====="
grep -E "Acc:|Edit:|F1@0.10:|F1@0.25:|F1@0.50:" "${LOG_DIR}/eval_${DATASET}_split${SPLIT}.log"

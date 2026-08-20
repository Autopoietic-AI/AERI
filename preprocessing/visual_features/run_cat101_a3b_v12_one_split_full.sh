#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:?Usage: bash m10/a3b_v12/run_cat101_a3b_v12_one_split_full.sh <split> [frame_epochs]}"
FRAME_EPOCHS="${2:-2}"

ROOT=${ROOT:-/path/to/MS-TCN2}
cd "${ROOT}"

FRAME_BATCH=16
FRAME_LR=3e-5
MAX_FRAMES=80
VAL_MAX_FRAMES=40

DATASET="cataract101_m10_a3b_v12_dualgate_s${SPLIT}_best"
CKPT="checkpoints/m10/a3b_v12/cat101_a3b_v12_dualgate_split${SPLIT}_best.pth"

MSTCN_MAIN="${ROOT}/m8/main_m8_v12_mstcnpp.py"
MSTCN_EPOCHS=100
MSTCN_LR=0.0005

LOG_DIR="${ROOT}/logs/m10"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "[M10-A3b] Full pipeline for split ${SPLIT}"
echo "FRAME_EPOCHS=${FRAME_EPOCHS}"
echo "DATASET=${DATASET}"
echo "CKPT=${CKPT}"
echo "============================================================"

echo ""
echo "===== STEP 1: Train A3b-v12 frame encoder ====="
python -u m10/a3b_v12/train_cat101_a3b_v12_dualgate_framecls.py \
  --split "${SPLIT}" \
  --epochs "${FRAME_EPOCHS}" \
  --batch_size "${FRAME_BATCH}" \
  --max_frames_per_class_per_video "${MAX_FRAMES}" \
  --val_max_frames_per_class_per_video "${VAL_MAX_FRAMES}" \
  --lr "${FRAME_LR}" \
  --no_amp \
  --print_every 50 \
  2>&1 | tee "${LOG_DIR}/train_cat101_a3b_v12_split${SPLIT}_ep${FRAME_EPOCHS}_bs${FRAME_BATCH}_lr3e5_noamp.log"

echo ""
echo "===== STEP 2: Check checkpoint ====="
if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] checkpoint not found: ${CKPT}"
  exit 1
fi
ls -lh "${CKPT}"

echo ""
echo "===== STEP 3: Extract 384D features ====="
python -u m10/a3b_v12/extract_cat101_a3b_v12_dualgate_features.py \
  --out_dataset "${DATASET}" \
  --ckpt "${CKPT}" \
  --batch_size 32 \
  2>&1 | tee "${LOG_DIR}/extract_${DATASET}_all.log"

echo ""
echo "===== STEP 4: Check extracted features ====="
python - <<PY
import numpy as np
from pathlib import Path

D = Path("${ROOT}") / "data" / "${DATASET}"
files = sorted((D / "features").glob("case_*.npy"))
bad = []
for p in files:
    arr = np.load(p, mmap_mode="r")
    gt = D / "groundTruth" / f"{p.stem}.txt"
    gt_len = sum(1 for _ in open(gt, encoding="utf-8"))
    if arr.shape != (384, gt_len):
        bad.append((p.name, arr.shape, gt_len))

print("dataset:", D)
print("feature files:", len(files))
print("bad count:", len(bad))
print("bad samples:", bad[:10])

if len(files) != 101 or bad:
    raise SystemExit("Feature check failed.")
PY

echo ""
echo "===== STEP 5: Train MS-TCN++ ====="
python -u "${MSTCN_MAIN}" \
  --action train \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --num_epochs "${MSTCN_EPOCHS}" \
  --learning_rate "${MSTCN_LR}" \
  --num_layers_PG 11 \
  --num_layers_R 10 \
  --num_R 4 \
  --num_f_maps 64 \
  --smooth_w 0.15 \
  --expected_dim 384 \
  2>&1 | tee "${LOG_DIR}/train_${DATASET}_split${SPLIT}.log"

echo ""
echo "===== STEP 6: Predict MS-TCN++ ====="
python -u "${MSTCN_MAIN}" \
  --action predict \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --num_epochs "${MSTCN_EPOCHS}" \
  --learning_rate "${MSTCN_LR}" \
  --num_layers_PG 11 \
  --num_layers_R 10 \
  --num_R 4 \
  --num_f_maps 64 \
  --smooth_w 0.15 \
  --expected_dim 384 \
  2>&1 | tee "${LOG_DIR}/predict_${DATASET}_split${SPLIT}.log"

echo ""
echo "===== STEP 7: Prepare eval result path ====="
REAL_RESULT_DIR="${ROOT}/results/tecno_${DATASET}/split_${SPLIT}"
EVAL_PARENT="${ROOT}/results/${DATASET}"
EVAL_DIR="${EVAL_PARENT}/split_${SPLIT}"

if [ ! -d "${REAL_RESULT_DIR}" ]; then
  echo "[ERROR] prediction dir not found: ${REAL_RESULT_DIR}"
  echo "Existing related result dirs:"
  find "${ROOT}/results" -type d | grep "${DATASET}" || true
  exit 1
fi

mkdir -p "${EVAL_PARENT}"
rm -rf "${EVAL_DIR}"
ln -s "$(realpath ${REAL_RESULT_DIR})" "${EVAL_DIR}"
ls -lh "${EVAL_PARENT}"

echo ""
echo "===== STEP 8: Eval ====="
python "${ROOT}/eval.py" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  2>&1 | tee "${LOG_DIR}/eval_${DATASET}_split${SPLIT}.log"

echo ""
echo "===== FINAL RESULT split ${SPLIT} ====="
grep -E "Acc:|Edit:|F1@0.10:|F1@0.25:|F1@0.50:" "${LOG_DIR}/eval_${DATASET}_split${SPLIT}.log"

echo ""
echo "[DONE] M10-A3b split ${SPLIT} full pipeline finished."

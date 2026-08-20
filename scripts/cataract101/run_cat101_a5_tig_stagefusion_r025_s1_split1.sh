#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/path/to/MS-TCN2}
DATASET=cataract101_m10_a5_tig_stagefusion_r025_s1_best
SPLIT=1

MAIN_PY=${ROOT}/m10/a5_tig_stagefusion/main_cat101_a5_tig_stagefusion.py
MODEL_PY=${ROOT}/m10/a5_tig_stagefusion/model_a5_tig_stagefusion.py
MASKSHAPE_DIR=${ROOT}/data/cataract101_m10_a5_maskshape_assets_v1_safe/npz
LOG_DIR=${ROOT}/logs/m10

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

# Keep fixed base gate as the residual anchor.
# v7 StageFusion will mix base_gate and dynamic_gate internally.
export A5_DFUSION_FIXED_GATE=1

# Stage-aware Dynamic Fusion
# 0.5 = half original fixed gate, half dynamic stage-aware gate.
# 如果不稳定，下一轮可以试 0.25。
export A5_STAGE_GATE_RESIDUAL=0.25

# Memory soft selection bandwidth.
export A5_MEMORY_SIGMA=0.22

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "[M10-A5-TIG-Dfusion-v7 Stage-aware Dynamic Fusion] split ${SPLIT}"
echo "ROOT=${ROOT}"
echo "DATASET=${DATASET}"
echo "MAIN_PY=${MAIN_PY}"
echo "MODEL_PY=${MODEL_PY}"
echo "MASKSHAPE_DIR=${MASKSHAPE_DIR}"
echo "A5_TIG_DILATED_OFFSETS=${A5_TIG_DILATED_OFFSETS}"
echo "A5_STAGE_GATE_RESIDUAL=${A5_STAGE_GATE_RESIDUAL}"
echo "A5_MEMORY_SIGMA=${A5_MEMORY_SIGMA}"
echo "============================================================"

echo
echo "===== STEP 0: Compile check ====="
python -m py_compile "${MODEL_PY}"
python -m py_compile "${MAIN_PY}"

echo
echo "===== STEP 1: Check dataset / assets ====="
ls -lh "${ROOT}/data/${DATASET}"
echo
echo "[features]"
find "${ROOT}/data/${DATASET}/features" -maxdepth 1 -type f | sort | head -5
echo
echo "[geo_assets]"
find "${ROOT}/data/${DATASET}/geo_assets" -maxdepth 1 -type f | sort | head -5
echo
echo "[maskshape]"
find "${MASKSHAPE_DIR}" -maxdepth 1 -type f | sort | head -5

echo
echo "===== STEP 2: Quick shape check ====="
python - <<PY
import numpy as np
from pathlib import Path

D = Path("${ROOT}") / "data" / "${DATASET}"
M = Path("${MASKSHAPE_DIR}")

for case in ["case_269", "case_270", "case_738"]:
    print("\\n", case)

    feat_p = D / "features" / f"{case}.npy"
    geo_p = D / "geo_assets" / f"{case}.npz"
    mask_p = M / f"{case}.npz"

    print("feat exists:", feat_p.exists(), feat_p)
    print("geo exists:", geo_p.exists(), geo_p)
    print("maskshape exists:", mask_p.exists(), mask_p)

    if feat_p.exists():
        f = np.load(feat_p, mmap_mode="r")
        print("feat:", f.shape, f.dtype)

    if geo_p.exists():
        a = np.load(geo_p, allow_pickle=True)
        print("geo:", a["geo"].shape, a["geo"].dtype)
        print("traj:", a["traj"].shape, a["traj"].dtype)
        print("aux:", a["aux"].shape, a["aux"].dtype)

    if mask_p.exists():
        q = np.load(mask_p, allow_pickle=True)
        print("mask_shape:", q["mask_shape"].shape, q["mask_shape"].dtype)
        print("mask_shape_valid:", q["mask_shape_valid"].shape, q["mask_shape_valid"].dtype)
PY

echo
echo "===== STEP 3: Train A5-TIG StageFusion ====="

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
echo "===== STEP 4: Predict ====="

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
echo "===== STEP 5: Eval ====="

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
echo "===== FINAL RESULT A5-TIG-Dfusion-v7 StageFusion split ${SPLIT} ====="
grep -E "Acc:|Edit:|F1@0.10:|F1@0.25:|F1@0.50:" "${LOG_DIR}/eval_${DATASET}_split${SPLIT}.log"

echo
echo "===== Gate log preview ====="
grep -E "gate_mean|gv=|ge=|ga=|gstd=" "${LOG_DIR}/train_${DATASET}_split${SPLIT}.log" | head -20 || true
grep -E "gate_mean|gv=|ge=|ga=|gstd=" "${LOG_DIR}/train_${DATASET}_split${SPLIT}.log" | tail -20 || true

echo
echo "===== Done ====="

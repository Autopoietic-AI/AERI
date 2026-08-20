#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/path/to/MS-TCN2}

SRC_DATA=${ROOT}/data/sics155_100_geoig_r025
A4_DIR=${ROOT}/data/sics155_100_a4_geo_assets_sam2_yolo_tiptrack_cons_compat/npz
MASKSHAPE_DIR=${ROOT}/data/sics155_100_a5_maskshape_assets_v1_safe/npz

DATASET=sics155_100_a5_visualonly_seed2026_trainval
DATA_DIR=${ROOT}/data/${DATASET}

SPLITS="${SPLITS:-2 3 4}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-2026}"

CAND1=${ROOT}/m10/a5_tig_stagefusion_r025_ease_seeded
CAND2=${ROOT}/m10/a5_tig_stagefusion_r025_rsens_seeded

if [ -f "${CAND1}/model_a5_tig_stagefusion.py" ] && grep -q "A5_ABLATE_GEO" "${CAND1}/model_a5_tig_stagefusion.py"; then
  CODE_DIR=${CAND1}
elif [ -f "${CAND2}/model_a5_tig_stagefusion.py" ] && grep -q "A5_ABLATE_GEO" "${CAND2}/model_a5_tig_stagefusion.py"; then
  CODE_DIR=${CAND2}
else
  echo "[ERROR] Cannot find model code supporting A5_ABLATE_* switches."
  grep -RIn "A5_ABLATE_GEO\|A5_ABLATE_TRAJ\|A5_ABLATE_MASKDYN" ${ROOT}/m10/a5_tig_stagefusion* || true
  exit 1
fi

MAIN_PY=${CODE_DIR}/main_cat101_a5_tig_stagefusion.py
MODEL_PY=${CODE_DIR}/model_a5_tig_stagefusion.py

LOG_DIR=${ROOT}/logs/m10_sics155_100_visualonly/${DATASET}
mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export A5_GLOBAL_SEED=${SEED}

export A5_DFUSION_F_DIM=48
export A5_DFUSION_E_LOCAL_DIM=12
export A5_TIG_E_TEMP_DIM=12
export A5_DFUSION_A_DIM=4
export A5_DFUSION_HIDDEN=128

export A5_TIG_Q_DIM=64
export A5_TIG_WINDOW=9
export A5_TIG_DILATED_OFFSETS="-16,-8,-4,-2,0,2,4,8,16"

export A5_DFUSION_FIXED_GATE=1
export A5_STAGE_GATE_RESIDUAL=0.0
export A5_MEMORY_SIGMA=0.22

export A5_ABLATE_GEO=1
export A5_ABLATE_TRAJ=1
export A5_ABLATE_MASKDYN=1
export A5_ABLATE_REL=1
export A5_ABLATE_AUX=1
export A5_ABLATE_DISTSPEED=1

unset A5_ABLATE_TRAJ_DIMS || true
unset A5_TIG_MEAN_POOL || true
unset A5_TIG_DISABLE_PROGRESS || true
unset A5_TIG_REL_BIAS_LAMBDA || true

echo "============================================================"
echo "[SICS155_100 Visual-only trainval multisplit]"
echo "DATASET=${DATASET}"
echo "SPLITS=${SPLITS}"
echo "EPOCHS=${EPOCHS}"
echo "SEED=${SEED}"
echo "CODE_DIR=${CODE_DIR}"
echo "============================================================"

echo
echo "===== Build trainval dataset alias ====="
rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}/splits"

cp -f "${SRC_DATA}/mapping.txt" "${DATA_DIR}/mapping.txt"
ln -sfn "${SRC_DATA}/features" "${DATA_DIR}/features"
ln -sfn "${SRC_DATA}/groundTruth" "${DATA_DIR}/groundTruth"
ln -sfn "${A4_DIR}" "${DATA_DIR}/geo_assets"

for sp in 1 2 3 4; do
  cp -f "${SRC_DATA}/splits/trainval.split${sp}.bundle" "${DATA_DIR}/splits/train.split${sp}.bundle"
  cp -f "${SRC_DATA}/splits/test.split${sp}.bundle" "${DATA_DIR}/splits/test.split${sp}.bundle"
  cp -f "${SRC_DATA}/splits/val.split${sp}.bundle" "${DATA_DIR}/splits/val.split${sp}.bundle"
  cp -f "${SRC_DATA}/splits/trainval.split${sp}.bundle" "${DATA_DIR}/splits/trainval.split${sp}.bundle"
done

python -m py_compile "${MODEL_PY}"
python -m py_compile "${MAIN_PY}"

for SPLIT in ${SPLITS}; do
  echo
  echo "============================================================"
  echo "[Visual-only split ${SPLIT}]"
  echo "train=$(wc -l < ${DATA_DIR}/splits/train.split${SPLIT}.bundle)"
  echo "test=$(wc -l < ${DATA_DIR}/splits/test.split${SPLIT}.bundle)"
  echo "============================================================"

  rm -rf "${ROOT}/models/tecno_${DATASET}/split_${SPLIT}"
  rm -rf "${ROOT}/results/tecno_${DATASET}/split_${SPLIT}"
  rm -rf "${ROOT}/results/${DATASET}/split_${SPLIT}"
  rm -rf "${ROOT}/artifacts/m10_a4_geoig_gate/${DATASET}/split_${SPLIT}"

  echo
  echo "===== Train visual-only split ${SPLIT} ====="
  python "${MAIN_PY}" \
    --action train \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --num_epochs "${EPOCHS}" \
    --learning_rate 0.0005 \
    --num_f_maps 64 \
    --smooth_w 0.15 \
    --expected_dim 384 \
    --m8_feature_dir "${DATA_DIR}/features" \
    --m7_asset_dir "${DATA_DIR}/geo_assets" \
    --maskshape_dir "${MASKSHAPE_DIR}" \
    2>&1 | tee "${LOG_DIR}/train_split${SPLIT}_e${EPOCHS}.log"

  echo
  echo "===== Predict visual-only split ${SPLIT} ====="
  python "${MAIN_PY}" \
    --action predict \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --num_epochs "${EPOCHS}" \
    --learning_rate 0.0005 \
    --num_f_maps 64 \
    --smooth_w 0.15 \
    --expected_dim 384 \
    --m8_feature_dir "${DATA_DIR}/features" \
    --m7_asset_dir "${DATA_DIR}/geo_assets" \
    --maskshape_dir "${MASKSHAPE_DIR}" \
    2>&1 | tee "${LOG_DIR}/predict_split${SPLIT}_e${EPOCHS}.log"

  echo
  echo "===== Eval visual-only raw split ${SPLIT} ====="
  EVAL_PARENT="${ROOT}/results/${DATASET}"
  EVAL_DIR="${EVAL_PARENT}/split_${SPLIT}"
  REAL_RESULT_DIR="${ROOT}/results/tecno_${DATASET}/split_${SPLIT}"

  mkdir -p "${EVAL_PARENT}"
  rm -rf "${EVAL_DIR}"
  ln -s "$(realpath "${REAL_RESULT_DIR}")" "${EVAL_DIR}"

  python "${ROOT}/eval.py" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    2>&1 | tee "${LOG_DIR}/eval_split${SPLIT}_e${EPOCHS}.log"

  echo
  echo "===== FINAL visual-only raw split ${SPLIT} ====="
  grep -E "Acc:|Edit:|F1@0.10:|F1@0.25:|F1@0.50:" "${LOG_DIR}/eval_split${SPLIT}_e${EPOCHS}.log"
done

#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/path/to/MS-TCN2}

SRC_DATA=${ROOT}/data/sics155_100_geoig_r025
A4_DIR=${ROOT}/data/sics155_100_a4_geo_assets_sam2_yolo_tiptrack_cons_compat/npz
MASKSHAPE_DIR=${ROOT}/data/sics155_100_a5_maskshape_assets_v1_safe/npz

DATASET=sics155_100_a5_tig_stagefusion_r025_rsens_seed2026_trainval
DATA_DIR=${ROOT}/data/${DATASET}

SPLITS=${SPLITS:-"1 2 3 4"}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-2026}

CODE_DIR=${ROOT}/m10/a5_tig_stagefusion_r025_rsens_seeded
MAIN_PY=${CODE_DIR}/main_cat101_a5_tig_stagefusion.py
MODEL_PY=${CODE_DIR}/model_a5_tig_stagefusion.py

LOG_DIR=${ROOT}/logs/m10_sics155_100_rsens/${DATASET}
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
export A5_MEMORY_SIGMA=0.22
export A5_STAGE_GATE_RESIDUAL=0.25

unset A5_ABLATE_GEO || true
unset A5_ABLATE_TRAJ || true
unset A5_ABLATE_MASKDYN || true
unset A5_ABLATE_REL || true
unset A5_ABLATE_AUX || true
unset A5_ABLATE_DISTSPEED || true
unset A5_ABLATE_TRAJ_DIMS || true
unset A5_TIG_MEAN_POOL || true
unset A5_TIG_DISABLE_PROGRESS || true
unset A5_TIG_REL_BIAS_LAMBDA || true

echo "============================================================"
echo "[SICS155_100 trainval StageFusion-r0.25 seed2026]"
echo "DATASET=${DATASET}"
echo "SPLITS=${SPLITS}"
echo "EPOCHS=${EPOCHS}"
echo "CODE_DIR=${CODE_DIR}"
echo "A5_GLOBAL_SEED=${A5_GLOBAL_SEED}"
echo "A5_STAGE_GATE_RESIDUAL=${A5_STAGE_GATE_RESIDUAL}"
echo "============================================================"

echo
echo "===== Build dataset alias with train=trainval ====="
rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}/splits"

cp -f "${SRC_DATA}/mapping.txt" "${DATA_DIR}/mapping.txt"
ln -sfn "${SRC_DATA}/features" "${DATA_DIR}/features"
ln -sfn "${SRC_DATA}/groundTruth" "${DATA_DIR}/groundTruth"
ln -sfn "${A4_DIR}" "${DATA_DIR}/geo_assets"

# Key point:
# train.splitX.bundle is replaced by trainval.splitX.bundle.
for sp in 1 2 3 4; do
  cp -f "${SRC_DATA}/splits/trainval.split${sp}.bundle" "${DATA_DIR}/splits/train.split${sp}.bundle"
  cp -f "${SRC_DATA}/splits/test.split${sp}.bundle" "${DATA_DIR}/splits/test.split${sp}.bundle"
  cp -f "${SRC_DATA}/splits/val.split${sp}.bundle" "${DATA_DIR}/splits/val.split${sp}.bundle"
  cp -f "${SRC_DATA}/splits/trainval.split${sp}.bundle" "${DATA_DIR}/splits/trainval.split${sp}.bundle"
done

echo
echo "===== Check split sizes ====="
for sp in 1 2 3 4; do
  echo "split ${sp}: train=$(wc -l < ${DATA_DIR}/splits/train.split${sp}.bundle) test=$(wc -l < ${DATA_DIR}/splits/test.split${sp}.bundle)"
done

python -m py_compile "${MODEL_PY}"
python -m py_compile "${MAIN_PY}"

grep -n "A5_TIG_REL_BIAS_LAMBDA\|A5_TIG_MEAN_POOL\|A5_TIG_DISABLE_PROGRESS" "${MODEL_PY}" \
  && { echo "[ERROR] model contains W-TIG ablation patch"; exit 1; } \
  || echo "[OK] no W-TIG ablation patch found"

for SPLIT in ${SPLITS}; do
  echo
  echo "################################################################"
  echo "### RUN trainval split ${SPLIT}"
  echo "################################################################"

  MODEL_DIR="${ROOT}/models/tecno_${DATASET}/split_${SPLIT}"
  RESULT_DIR="${ROOT}/results/tecno_${DATASET}/split_${SPLIT}"
  EVAL_PARENT="${ROOT}/results/${DATASET}"
  EVAL_DIR="${EVAL_PARENT}/split_${SPLIT}"

  echo
  echo "===== Train split ${SPLIT} ====="
  rm -rf "${MODEL_DIR}"

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
  echo "===== Predict split ${SPLIT} ====="
  rm -rf "${RESULT_DIR}"
  rm -rf "${ROOT}/artifacts/m10_a4_geoig_gate/${DATASET}/split_${SPLIT}"

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
  echo "===== Eval split ${SPLIT} ====="
  mkdir -p "${EVAL_PARENT}"
  rm -rf "${EVAL_DIR}"

  ln -s "$(realpath "${RESULT_DIR}")" "${EVAL_DIR}"

  python "${ROOT}/eval.py" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    2>&1 | tee "${LOG_DIR}/eval_split${SPLIT}_e${EPOCHS}.log"

  echo
  echo "===== RESULT split ${SPLIT} ====="
  grep -E "Acc:|Edit:|F1@0.10:|F1@0.25:|F1@0.50:" "${LOG_DIR}/eval_split${SPLIT}_e${EPOCHS}.log"
done

echo
echo "===== Collect summary ====="
python - <<PY
import re
from pathlib import Path
import numpy as np

log_dir = Path("${LOG_DIR}")
epochs = "${EPOCHS}"
metrics = ["Acc", "Edit", "F1@0.10", "F1@0.25", "F1@0.50"]

rows = []
for sp in [1,2,3,4]:
    p = log_dir / f"eval_split{sp}_e{epochs}.log"
    if not p.exists():
        print("[MISS]", p)
        continue
    text = p.read_text(errors="ignore")
    vals = {}
    for m in metrics:
        mm = re.search(rf"{re.escape(m)}:\\s*([0-9.]+)", text)
        if mm:
            vals[m] = float(mm.group(1))
    if len(vals) == len(metrics):
        rows.append((sp, vals))
        print(f"split{sp}: " + " | ".join(f"{m}={vals[m]:.4f}" for m in metrics))

txt = log_dir / "summary_4fold_trainval.txt"
with txt.open("w") as f:
    for sp, vals in rows:
        f.write(f"split{sp}: " + " | ".join(f"{m}={vals[m]:.4f}" for m in metrics) + "\\n")
    f.write("\\nMean ± Std\\n")
    print("\\n===== Mean ± Std =====")
    for m in metrics:
        arr = np.array([v[m] for _, v in rows], dtype=float)
        line = f"{m}: {arr.mean():.4f} ± {arr.std(ddof=0):.4f}"
        print(line)
        f.write(line + "\\n")

print("[SAVE]", txt)
PY

echo
echo "===== DONE trainval 4-fold ====="

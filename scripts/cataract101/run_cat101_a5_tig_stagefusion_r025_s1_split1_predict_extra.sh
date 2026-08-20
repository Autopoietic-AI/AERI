#!/usr/bin/env bash
set -euo pipefail

cd "${ROOT:-/path/to/MS-TCN2}"

export A5_STAGE_GATE_RESIDUAL=0.25
export A5_DFUSION_FIXED_GATE=1
export A5_TIG_DILATED_OFFSETS="-16,-8,-4,-2,0,2,4,8,16"

DATASET="cataract101_m10_a5_tig_stagefusion_r025_s1_best"
SPLIT=1
FEATURE_DIR="${ROOT:-/path/to/MS-TCN2}/data/${DATASET}/features"
GEO_DIR="${ROOT:-/path/to/MS-TCN2}/data/${DATASET}/geo_assets"

echo "===== Predict-only StageFusion-r0.25 split1 with extra dump ====="
echo "DATASET=${DATASET}"
echo "SPLIT=${SPLIT}"
echo "FEATURE_DIR=${FEATURE_DIR}"
echo "GEO_DIR=${GEO_DIR}"
echo "NOTE: this is predict-only, num_epochs only selects epoch-100.model"

python m10/a5_tig_stagefusion/main_cat101_a5_tig_stagefusion.py \
  --action predict \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --m8_feature_dir "${FEATURE_DIR}" \
  --m7_asset_dir "${GEO_DIR}" \
  --num_epochs 100 \
  --num_stages 4 \
  --num_layers 10 \
  --num_f_maps 64 \
  --main_dim 256 \
  --guide_dim 64 \
  --beta_max 0.15 \
  --gate_l1 0.001 \
  --smooth_w 0.25 \
  --expected_dim 384

# Cataract101 Pipeline

This document describes the Cataract101 reproduction pipeline for GeoIG-StageFusion-r0.25.

## 1. Overview

Cataract101 is the main dataset used for method validation, component analysis, and mechanism visualization.

The full pipeline is:

    visual features
      + YOLO detection assets
      + SAM2 masks
      + EASE / GeoIG structured evidence
      + maskshape features
      -> GeoIG-StageFusion-r0.25
      -> MS-TCN++ temporal decoder
      -> phase prediction timeline

## 2. Canonical model

The canonical implementation is:

    src/geoig_stagefusion/main_cat101_a5_tig_stagefusion.py
    src/geoig_stagefusion/model_a5_tig_stagefusion.py

The released files are copied from:

    m10/a5_tig_stagefusion_r025_rsens_seeded/

The model architecture is identical to the final Cataract101 StageFusion implementation. The seeded version adds deterministic seed control.

Main configuration:

    A5_GLOBAL_SEED=2026
    A5_STAGE_GATE_RESIDUAL=0.25
    A5_TIG_DILATED_OFFSETS="-16,-8,-4,-2,0,2,4,8,16"

## 3. Expected data layout

A typical Cataract101 StageFusion-ready dataset should contain:

    data/<cat101_dataset_name>/
      features/
      groundTruth/
      mapping.txt
      splits/

The GeoIG and maskshape assets are usually referenced separately by script arguments or environment variables:

    data/<cat101_geo_asset_dir>/npz/*.npz
    data/<cat101_maskshape_asset_dir>/npz/*.npz

Typical important assets:

    features/*.npy
    groundTruth/*.txt
    mapping.txt
    splits/train.splitX.bundle
    splits/test.splitX.bundle
    GeoIG npz assets
    maskshape npz assets

## 4. Preprocessing steps

### 4.1 Visual features

Relevant scripts:

    preprocessing/visual_features/train_cat101_a3b_v12_dualgate_framecls.py
    preprocessing/visual_features/extract_cat101_a3b_v12_dualgate_features.py
    preprocessing/visual_features/extract_cat101_rgb2048.py
    preprocessing/visual_features/extract_cat101_x5_nfnet3072.py

The final A5 StageFusion setting expects 384-dimensional features.

### 4.2 YOLO detection assets

Relevant script:

    preprocessing/yolo_detection/extract_cat101_yolo_npz_assets_v2_multiinst.py

Expected output:

    data/cataract101_yolo_b2_multiinst_assets/npz/*.npz

### 4.3 SAM2 masks

Relevant script:

    preprocessing/sam2_masks/build_cat101_sam2_masks_full_packed.py

Expected output:

    data/cataract101_sam2_masks_full/npz/*.npz

### 4.4 EASE / GeoIG structured evidence

Relevant scripts:

    preprocessing/ease_assets/build_cat101_a4_geo_assets_yolo_tiptrack_cons.py
    preprocessing/ease_assets/build_cat101_a4_geo_assets_hybrid_tip_center.py
    preprocessing/ease_assets/make_a4_v1c_cons_compat_assets.py

Expected output:

    data/cataract101_a4_geo_assets_sam2_yolo_tiptrack_cons_compat/npz/*.npz

### 4.5 Maskshape assets

Relevant script:

    preprocessing/maskshape/build_cat101_maskshape_assets_v1.py

Expected output:

    data/cataract101_a5_maskshape_assets_v1_safe/npz/*.npz

## 5. Main four-fold training and prediction

Main script:

    scripts/cataract101/run_cat101_a5_tig_stagefusion_r025_multisplit.sh

Original source:

    m10/a5_tig_stagefusion/run_cat101_a5_tig_stagefusion_r025_multisplit.sh

Usage pattern:

    cd /path/to/MS-TCN2
    bash scripts/cataract101/run_cat101_a5_tig_stagefusion_r025_multisplit.sh

This script trains and evaluates GeoIG-StageFusion-r0.25 across Cataract101 splits.

## 6. Single-split quick reproduction

Single-split script:

    scripts/cataract101/run_cat101_a5_tig_stagefusion_r025_s1_split1.sh

Usage pattern:

    cd /path/to/MS-TCN2
    bash scripts/cataract101/run_cat101_a5_tig_stagefusion_r025_s1_split1.sh

This is useful for quick sanity checking before running all splits.

## 7. Extra prediction and visualization assets

Extra prediction script:

    scripts/cataract101/run_cat101_a5_tig_stagefusion_r025_s1_split1_predict_extra.sh

This script is used to export additional prediction artifacts for visualization and mechanism analysis, such as:

    prediction timeline
    target labels
    TIG attention
    StageFusion gate
    reliability curves

## 8. Component masking

Component masking script:

    scripts/cataract101/run_r025_component_masking_md.sh

Ablation entry:

    src/geoig_stagefusion/main_cat101_a5_tig_stagefusion_ablation.py

This is used for checkpoint-controlled component masking. It evaluates the contribution of evidence groups and fusion branches without retraining a new model.

## 9. Expected main result

GeoIG-StageFusion-r0.25 on Cataract101:

| Metric | Mean ± Std |
|---|---:|
| Acc | 91.4460 ± 4.3981 |
| Edit | 87.1281 ± 7.8541 |
| F1@0.10 | 90.2695 ± 6.6507 |
| F1@0.25 | 88.9021 ± 8.0818 |
| F1@0.50 | 83.3764 ± 11.2133 |

## 10. Notes

The Cataract101 pipeline is the main validation track for the paper.

SICS155_100 is used as additional validation for robustness under more challenging manual small-incision cataract surgery videos.

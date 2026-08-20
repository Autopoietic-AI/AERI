# Dataset Notes

This document describes the dataset assumptions and expected layouts used by the GeoIG-StageFusion release.

## 1. Supported experiment tracks

This release is organized around two experiment tracks:

| Dataset | Role |
|---|---|
| Cataract101 | Main validation dataset for method comparison, component analysis, and mechanism visualization. |
| SICS155_100 | Additional validation dataset for robustness analysis under more fine-grained manual small-incision cataract surgery videos. |

## 2. Important release policy

This repository does not include:

| Data type | Included? |
|---|---|
| Raw videos | No |
| Extracted frames | No |
| Visual features | No |
| YOLO detection outputs | No |
| SAM2 masks | No |
| GeoIG / EASE assets | No |
| Maskshape assets | No |
| Model checkpoints | No |
| Prediction files | No |
| Training logs | No |

Users need to prepare these files separately.

## 3. Cataract101

Cataract101 is the main dataset used in the paper.

Expected StageFusion-ready layout:

    data/<cat101_dataset_name>/
      features/
      groundTruth/
      mapping.txt
      splits/

Typical files:

    data/<cat101_dataset_name>/features/*.npy
    data/<cat101_dataset_name>/groundTruth/*.txt
    data/<cat101_dataset_name>/mapping.txt
    data/<cat101_dataset_name>/splits/train.splitX.bundle
    data/<cat101_dataset_name>/splits/test.splitX.bundle

Additional asset directories:

    data/<cat101_yolo_asset_dir>/npz/*.npz
    data/<cat101_sam2_mask_dir>/npz/*.npz
    data/<cat101_geo_asset_dir>/npz/*.npz
    data/<cat101_maskshape_asset_dir>/npz/*.npz

## 4. SICS155_100

SICS155_100 denotes the 100-video SICS setting used in this project.

In the paper and release documents, describe it carefully as:

    SICS155_100 additional validation setting

or:

    the 100-video SICS155_100 setting used in our controlled experiments

Do not describe it as the complete SICS-155 dataset unless verified against the official release notes.

Expected StageFusion-ready layout:

    data/sics155_100_geoig_r025/
      features/
      groundTruth/
      mapping.txt
      splits/

Expected asset directories:

    data/sics155_100_a4_geo_assets_sam2_yolo_tiptrack_cons_compat/npz/*.npz
    data/sics155_100_a5_maskshape_assets_v1_safe/npz/*.npz

The trainval setting uses:

    train.splitX.bundle = trainval.splitX.bundle
    test.splitX.bundle = held-out test split

## 5. Phase label mapping

Each dataset should provide:

    mapping.txt

This file maps class indices to phase names.

The evaluation script expects prediction labels and ground-truth labels to match the names in mapping.txt.

## 6. Feature length consistency

For every video:

    number of feature time steps
    = number of ground-truth labels
    = number of GeoIG asset frames
    = number of maskshape frames

Before training, check sequence length consistency.

Recommended checking scripts:

    preprocessing/visual_features/step3_check_cat101_and_sics_features.py
    tools/diag_sics_clean_filtered.py

## 7. Expected feature dimension

The final A5 StageFusion setting expects:

    expected_dim = 384

This corresponds to the A3B v12 dual-gate feature setting used in the final SICS155_100 experiments.

## 8. Notes for GitHub users

Large generated files should be stored outside the repository.

Recommended external storage structure:

    /path/to/data/
    /path/to/checkpoints/
    /path/to/results/
    /path/to/logs/

Scripts should be adapted to point to the user's local data root.


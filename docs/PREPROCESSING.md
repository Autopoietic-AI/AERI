# Preprocessing Pipeline

This document describes the preprocessing flow used by GeoIG-StageFusion.

The full pipeline converts raw cataract surgery videos or extracted frames into the training-ready inputs required by the final StageFusion model.

## Overview

The preprocessing flow contains five major stages:

    Raw videos / frames
      -> visual feature extraction
      -> YOLO eye / instrument detection assets
      -> SAM2 packed masks
      -> EASE / GeoIG structured evidence assets
      -> maskshape assets
      -> StageFusion-ready dataset

The final model consumes:

    visual features
    GeoIG structured evidence assets
    auxiliary reliability cues
    trajectory cues
    maskshape features
    ground-truth phase labels
    split files

## 1. Visual feature extraction

Relevant folder:

    preprocessing/visual_features/

Important files:

| File | Role |
|---|---|
| train_cat101_a3b_v12_dualgate_framecls.py | Trains the A3B dual-gate frame-level feature extractor. |
| extract_cat101_a3b_v12_dualgate_features.py | Extracts Cataract101 A3B features. |
| extract_sics155_100_a3b_v12_dualgate_features.py | Extracts SICS155_100 A3B features. |
| extract_cat101_rgb2048.py | Extracts RGB2048 baseline features. |
| extract_cat101_x5_nfnet3072.py | Extracts NFNet3072 features. |
| step3_check_cat101_and_sics_features.py | Audits feature dimensions and sequence lengths. |

Expected output:

    data/<dataset_name>/features/*.npy

For the final A5 StageFusion setting, the expected feature dimension is 384.

## 2. YOLO detection assets

Relevant folder:

    preprocessing/yolo_detection/

Important files:

| File | Role |
|---|---|
| extract_cat101_yolo_npz_assets_v2_multiinst.py | Converts Cataract101 YOLO detections into multi-instance NPZ assets. |
| extract_sics155_100_yolo_npz_assets_v2_multiinst.py | Converts SICS155_100 YOLO detections into multi-instance NPZ assets. |
| visualize_cat101_yolo_samples.py | Visual sanity check for YOLO detections. |

Expected output:

    data/<dataset_name>_yolo_b2_multiinst_assets/npz/*.npz

These assets provide eye and instrument detections used by EASE / GeoIG asset construction.

## 3. SAM2 packed masks

Relevant folder:

    preprocessing/sam2_masks/

Important files:

| File | Role |
|---|---|
| build_cat101_sam2_masks_full_packed.py | Builds packed SAM2 masks for Cataract101. |
| build_sics155_100_sam2_masks_full_packed.py | Builds packed SAM2 masks for SICS155_100. |
| check_sam2_paths.py | Checks SAM2 path settings. |
| check_sam2_prereq.sh | Checks SAM2 prerequisites. |
| check_sam2_prereq_v2.sh | Updated prerequisite checker. |
| inspect_sam2_packed_mask_scale.py | Inspects packed mask scale and shape consistency. |
| vis_sam2_packed_mask_overlay.py | Visualizes packed masks over frames. |

Expected output:

    data/<dataset_name>_sam2_masks_full/npz/*.npz

SAM2 masks are used for eye / instrument mask cues and maskshape feature construction.

## 4. EASE / GeoIG structured evidence assets

Relevant folder:

    preprocessing/ease_assets/

Important files:

| File | Role |
|---|---|
| build_cat101_a4_geo_assets_yolo_tiptrack_cons.py | Builds Cataract101 A4 GeoIG assets from YOLO detections with conservative tip tracking. |
| build_sics155_100_a4_geo_assets_yolo_tiptrack_cons_compat.py | Builds SICS155_100 A4 GeoIG assets in a compatible format. |
| build_cat101_a4_geo_assets_hybrid_tip_center.py | Builds hybrid tip / center-proxy assets. |
| make_a4_v1c_cons_compat_assets.py | Converts or repairs assets into final compatibility format. |
| analyze_cat101_a4_traj_importance.py | Analyzes trajectory cue importance. |
| visualize_cat101_a4_geo_assets.py | Visualizes GeoIG assets. |
| visualize_cat101_a4_traj_dims.py | Visualizes trajectory dimensions. |

Expected output:

    data/<dataset_name>_a4_geo_assets_sam2_yolo_tiptrack_cons_compat/npz/*.npz

Main asset fields include:

    geo
    aux
    presence_mask
    cls_proxy
    cls_proxy_mask
    traj
    eye_center
    eye_radius
    inst_tips
    eye_valid
    inst_valid_any

These fields are converted into structured instrument-eye interaction evidence.

## 5. Maskshape assets

Relevant folder:

    preprocessing/maskshape/

Important files:

| File | Role |
|---|---|
| build_cat101_maskshape_assets_v1.py | Builds Cataract101 maskshape assets. |
| build_sics155_100_maskshape_assets_v1_safe.py | Builds SICS155_100 maskshape assets with safe validity handling. |
| make_maskshape_v1_clean.py | Cleans maskshape assets. |
| make_maskshape_v1_safe.py | Builds validity-aware safe maskshape assets. |
| plot_maskshape_timeline.py | Visualizes maskshape trajectories. |

Expected output:

    data/<dataset_name>_a5_maskshape_assets_v1_safe/npz/*.npz

Main fields include:

    mask_shape
    mask_shape_valid

Maskshape features are used as temporal mask-dynamic cues in the StageFusion branch.

## 6. StageFusion-ready dataset layout

A training-ready dataset should follow this layout:

    data/<stagefusion_dataset_name>/
      features/
      groundTruth/
      mapping.txt
      splits/
      geo_assets/
      maskshape_assets/    optional or externally referenced by script

For many internal scripts, feature and asset directories are supplied separately by command-line arguments or environment variables.

Typical required files:

    data/<dataset>/mapping.txt
    data/<dataset>/groundTruth/*.txt
    data/<dataset>/splits/train.splitX.bundle
    data/<dataset>/splits/test.splitX.bundle
    data/<dataset>/features/*.npy

## 7. Quality and sanity checks

Before training, check:

    Number of feature files matches number of videos.
    Feature dimension matches expected_dim.
    Feature temporal length matches groundTruth temporal length.
    GeoIG asset temporal length matches feature length.
    maskshape asset temporal length matches feature length.
    mapping.txt phase names match prediction labels.

Useful scripts:

    preprocessing/visual_features/step3_check_cat101_and_sics_features.py
    preprocessing/sam2_masks/inspect_sam2_packed_mask_scale.py
    preprocessing/ease_assets/visualize_cat101_a4_geo_assets.py
    preprocessing/maskshape/plot_maskshape_timeline.py

## 8. Notes

The release does not include generated preprocessing outputs.

Large generated files should not be committed to GitHub:

    raw videos
    extracted frames
    .npy features
    .npz assets
    SAM2 masks
    YOLO outputs
    checkpoints
    predictions
    logs

These should be stored outside the repository and referenced by configurable paths in scripts.


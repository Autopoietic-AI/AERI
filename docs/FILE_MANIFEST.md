> Historical GeoIG release documentation. Retained for provenance and existing utilities; see README.md, docs/CANONICAL_IMPLEMENTATION.md and docs/RELEASE_SCOPE.md for the current AERI source release. Historical result tables and fallback settings are not the current paper's result/protocol declaration.

# File Manifest

This document describes the role of each file in the GeoIG-StageFusion release.

## 1. Core model

| File | Description |
|---|---|
| src/geoig_stagefusion/model_a5_tig_stagefusion.py | Core implementation of GeoIG-StageFusion. It contains the structured interaction branch, W-TIG temporal evidence reasoning, RC-StageFusion, and MS-TCN++ temporal decoder. |
| src/geoig_stagefusion/main_cat101_a5_tig_stagefusion.py | Main training and prediction entry point. Despite the historical cat101 name, it can also run SICS155_100 by changing dataset and asset paths. |
| src/geoig_stagefusion/main_cat101_a5_tig_stagefusion_ablation.py | Component masking and ablation entry point for checkpoint-controlled analysis. |
| src/geoig_stagefusion/eval.py | Evaluation script for frame-wise accuracy, Edit score, and segmental F1@10/25/50. |

## 2. Visual feature preprocessing

| File | Description |
|---|---|
| preprocessing/visual_features/train_cat101_a3b_v12_dualgate_framecls.py | Trains the A3B dual-gate frame-level visual feature extractor. |
| preprocessing/visual_features/extract_cat101_a3b_v12_dualgate_features.py | Extracts Cataract101 A3B visual features. |
| preprocessing/visual_features/extract_sics155_100_a3b_v12_dualgate_features.py | Extracts SICS155_100 A3B visual features. |
| preprocessing/visual_features/extract_cat101_rgb2048.py | Extracts RGB 2048-dimensional baseline features for Cataract101. |
| preprocessing/visual_features/extract_cat101_x5_nfnet3072.py | Extracts NFNet-based 3072-dimensional features for Cataract101. |
| preprocessing/visual_features/run_cat101_a3b_v12_one_split_full.sh | Shell script for one-split A3B feature training and extraction. |
| preprocessing/visual_features/run_cat101_a3b_v12_s1_best_mstcnpp_split1.sh | Script for evaluating A3B features with MS-TCN++ on split 1. |
| preprocessing/visual_features/step3_check_cat101_and_sics_features.py | Audits extracted feature files, dimensions, and sequence lengths. |

## 3. YOLO detection assets

| File | Description |
|---|---|
| preprocessing/yolo_detection/extract_cat101_yolo_npz_assets_v2_multiinst.py | Converts Cataract101 YOLO eye and instrument detections into multi-instance NPZ assets. |
| preprocessing/yolo_detection/extract_sics155_100_yolo_npz_assets_v2_multiinst.py | Converts SICS155_100 YOLO detections into multi-instance NPZ assets. |
| preprocessing/yolo_detection/visualize_cat101_yolo_samples.py | Visualizes YOLO detections for sanity checking. |

## 4. SAM2 mask assets

| File | Description |
|---|---|
| preprocessing/sam2_masks/build_cat101_sam2_masks_full_packed.py | Builds packed SAM2 masks for Cataract101. |
| preprocessing/sam2_masks/build_sics155_100_sam2_masks_full_packed.py | Builds packed SAM2 masks for SICS155_100. |
| preprocessing/sam2_masks/check_sam2_paths.py | Checks SAM2 path configuration. |
| preprocessing/sam2_masks/check_sam2_prereq.sh | Checks SAM2 environment prerequisites. |
| preprocessing/sam2_masks/check_sam2_prereq_v2.sh | Updated SAM2 prerequisite checker. |
| preprocessing/sam2_masks/inspect_sam2_packed_mask_scale.py | Inspects packed SAM2 mask scale and shape consistency. |
| preprocessing/sam2_masks/vis_sam2_packed_mask_overlay.py | Visualizes packed SAM2 masks over frames. |

## 5. EASE / GeoIG structured evidence assets

| File | Description |
|---|---|
| preprocessing/ease_assets/build_cat101_a4_geo_assets_yolo_tiptrack_cons.py | Builds Cataract101 A4 GeoIG assets from YOLO detections with conservative tip tracking. |
| preprocessing/ease_assets/build_sics155_100_a4_geo_assets_yolo_tiptrack_cons_compat.py | Builds SICS155_100 A4 GeoIG assets in a Cataract101-compatible format. |
| preprocessing/ease_assets/build_cat101_a4_geo_assets_hybrid_tip_center.py | Builds hybrid tip and center-proxy GeoIG assets for Cataract101. |
| preprocessing/ease_assets/make_a4_v1c_cons_compat_assets.py | Converts or repairs A4 assets into the final compatibility format. |
| preprocessing/ease_assets/analyze_cat101_a4_traj_importance.py | Analyzes trajectory cue importance. |
| preprocessing/ease_assets/visualize_cat101_a4_geo_assets.py | Visualizes GeoIG structured evidence assets. |
| preprocessing/ease_assets/visualize_cat101_a4_traj_dims.py | Visualizes trajectory dimensions and dynamics. |

## 6. Maskshape assets

| File | Description |
|---|---|
| preprocessing/maskshape/build_cat101_maskshape_assets_v1.py | Builds Cataract101 mask-shape features. |
| preprocessing/maskshape/build_sics155_100_maskshape_assets_v1_safe.py | Builds SICS155_100 mask-shape features with safe validity handling. |
| preprocessing/maskshape/make_maskshape_v1_clean.py | Cleans maskshape assets. |
| preprocessing/maskshape/make_maskshape_v1_safe.py | Builds a safe version of maskshape assets with validity checks. |
| preprocessing/maskshape/plot_maskshape_timeline.py | Visualizes maskshape feature trajectories. |

## 7. Cataract101 experiment scripts

| File | Description |
|---|---|
| scripts/cataract101/run_cat101_a5_tig_stagefusion_r025_multisplit.sh | Main four-fold Cataract101 training and prediction script for GeoIG-StageFusion-r0.25. |
| scripts/cataract101/run_cat101_a5_tig_stagefusion_r025_s1_split1.sh | Single-split Cataract101 training and prediction script for quick reproduction. |
| scripts/cataract101/run_cat101_a5_tig_stagefusion_r025_s1_split1_predict_extra.sh | Extra prediction and artifact export script for diagnostics and visualization. |
| scripts/cataract101/run_r025_component_masking_md.sh | Runs checkpoint-controlled component masking for the r0.25 model. |

## 8. SICS155_100 experiment scripts

| File | Description |
|---|---|
| scripts/sics155/run_sics155_100_rsens_seed2026_r025_trainval_multisplit.sh | Four-fold StageFusion-r0.25 training and prediction on SICS155_100 trainval/test splits. |
| scripts/sics155/run_sics155_100_visualonly_trainval_multisplit_safe.sh | Four-fold visual-only expert training and prediction on SICS155_100. Used for reliability-aware fallback. |

## 9. Evaluation, smoothing, and fallback tools

| File | Description |
|---|---|
| tools/smooth_eval_dataset_v2.py | Applies one-frame de-spiking and evaluates smoothed predictions. |
| tools/make_sics_fallback_eval.py | Builds SICS fallback predictions by selecting visual-only predictions when structured evidence quality is low. |
| tools/collect_sics_4fold_metrics.py | Collects four-fold Acc, Edit, and F1 metrics for SICS experiments. |
| tools/sics_smoothing_sweep_v2.py | Sweeps different minimum segment lengths for smoothing analysis. |
| tools/diag_sics_clean_filtered.py | Clean per-video SICS diagnostic script. |
| tools/diag_sics_stagefusion_split1_v2.py | Earlier SICS diagnostic script for fragmentation and asset-quality analysis. |
| tools/build_cat101_4fold_surgeon_splits.py | Builds Cataract101 four-fold surgeon-level splits. |

## 10. Release notes

This repository does not include raw videos, extracted features, SAM2 masks, YOLO outputs, GeoIG assets, maskshape assets, model checkpoints, or generated predictions.

The canonical final model is GeoIG-StageFusion-r0.25.

For SICS155_100, the final reported method uses StageFusion + one-frame de-spiking + reliability-aware fallback.

# GeoIG-StageFusion

GeoIG-StageFusion is an eye-anchored structured interaction reasoning framework for cataract surgical phase recognition.

The method converts low-level visual, detection, segmentation, trajectory, and reliability cues into eye-anchored structured evidence, reasons over workflow-conditioned temporal interaction evidence, and injects the evidence into a visual-temporal phase recognition pipeline through conservative StageFusion.

## Main components

- EASE: Eye-Anchored Structured Evidence
- W-TIG: Workflow-conditioned Temporal Interaction Guidance
- RC-StageFusion-r0.25: Reliability-Constrained Conservative StageFusion
- MS-TCN++ temporal phase decoder

## Repository structure

| Folder | Description |
|---|---|
| src/geoig_stagefusion | Core model, training entry, ablation entry, and evaluation script. |
| preprocessing/visual_features | Visual feature extraction scripts. |
| preprocessing/yolo_detection | YOLO eye / instrument detection asset scripts. |
| preprocessing/sam2_masks | SAM2 mask construction and checking scripts. |
| preprocessing/ease_assets | EASE / GeoIG structured evidence asset construction scripts. |
| preprocessing/maskshape | Mask-shape feature construction scripts. |
| scripts/cataract101 | Cataract101 training and evaluation scripts. |
| scripts/sics155 | SICS155_100 training and robust inference scripts. |
| tools | Smoothing, fallback, diagnostic, split-building, and metric collection tools. |
| docs | Documentation for implementation, preprocessing, and experiments. |

## Canonical implementation

The released canonical implementation is:

    src/geoig_stagefusion/main_cat101_a5_tig_stagefusion.py
    src/geoig_stagefusion/model_a5_tig_stagefusion.py

These files are copied from:

    m10/a5_tig_stagefusion_r025_rsens_seeded/

The model file is identical to the original Cataract101 StageFusion implementation, while the seeded main entry adds deterministic seed control.

Main configuration:

    A5_STAGE_GATE_RESIDUAL=0.25
    A5_TIG_DILATED_OFFSETS="-16,-8,-4,-2,0,2,4,8,16"
    A5_GLOBAL_SEED=2026

## Datasets

This release is organized around two experiment tracks:

- Cataract101
- SICS155_100

Raw videos, extracted visual features, YOLO assets, SAM2 masks, GeoIG assets, maskshape assets, checkpoints, and generated predictions are not included in this repository.

Users should prepare the expected dataset layout following docs/PREPROCESSING.md and the dataset-specific pipeline documents.

## Main results

### Cataract101

GeoIG-StageFusion-r0.25:

| Metric | Mean ± Std |
|---|---:|
| Acc | 91.4460 ± 4.3981 |
| Edit | 87.1281 ± 7.8541 |
| F1@0.10 | 90.2695 ± 6.6507 |
| F1@0.25 | 88.9021 ± 8.0818 |
| F1@0.50 | 83.3764 ± 11.2133 |

### SICS155_100

Final robust inference variant:

    StageFusion + one-frame de-spiking + reliability-aware fallback

| Metric | Mean ± Std |
|---|---:|
| Acc | 77.3081 ± 1.1754 |
| Edit | 87.9522 ± 1.0700 |
| F1@0.10 | 86.0560 ± 1.2154 |
| F1@0.25 | 81.2107 ± 1.3959 |
| F1@0.50 | 68.4079 ± 1.9326 |

## Important documents

| Document | Description |
|---|---|
| docs/CANONICAL_IMPLEMENTATION.md | Clarifies the exact final model implementation. |
| docs/FILE_MANIFEST.md | Explains the role of every released file. |
| docs/PREPROCESSING.md | Describes the preprocessing flow from raw data to training-ready assets. |

## Notes

This repository is a cleaned research release. It keeps the final reproducible pipeline and selected preprocessing scripts, but does not include historical experimental branches or large generated assets.

The SICS155_100 reliability-aware fallback is not a separately trained model. It is an inference-time selection strategy triggered by input-side structured evidence quality.


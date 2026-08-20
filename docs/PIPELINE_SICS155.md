# SICS155_100 Pipeline

This document describes the SICS155_100 additional validation pipeline.

## 1. Overview

SICS155_100 is used as additional validation for a more challenging manual small-incision cataract surgery setting.

Compared with Cataract101, SICS155_100 contains more fine-grained phases and larger structured-evidence quality variation.

The final SICS155_100 reported method uses:

    StageFusion expert
      + one-frame de-spiking
      + reliability-aware fallback

The fallback is not a separate training model. It is an inference-time prediction selection strategy based on input-side structured evidence quality.

## 2. Canonical model

The same canonical model file is used:

    src/geoig_stagefusion/model_a5_tig_stagefusion.py

The model supports ablation switches:

    A5_ABLATE_GEO
    A5_ABLATE_TRAJ
    A5_ABLATE_MASKDYN
    A5_ABLATE_REL
    A5_ABLATE_AUX
    A5_ABLATE_DISTSPEED
    A5_ABLATE_TRAJ_DIMS

Therefore, the same model file supports both:

    StageFusion expert
    Visual-only expert

## 3. Expected data layout

The main SICS feature dataset is:

    data/sics155_100_geoig_r025/

Expected structure:

    data/sics155_100_geoig_r025/
      features/
      groundTruth/
      mapping.txt
      splits/

Expected assets:

    data/sics155_100_a4_geo_assets_sam2_yolo_tiptrack_cons_compat/npz/*.npz
    data/sics155_100_a5_maskshape_assets_v1_safe/npz/*.npz

The trainval split setting uses:

    train.splitX.bundle = trainval.splitX.bundle
    test.splitX.bundle = test split

## 4. StageFusion expert

Main script:

    scripts/sics155/run_sics155_100_rsens_seed2026_r025_trainval_multisplit.sh

Original source:

    m10/a5_tig_stagefusion_r025_rsens_seeded/run_sics155_100_rsens_seed2026_r025_trainval_multisplit.sh

Usage pattern:

    cd /path/to/MS-TCN2
    SPLITS="1 2 3 4" EPOCHS="100" bash scripts/sics155/run_sics155_100_rsens_seed2026_r025_trainval_multisplit.sh

Main configuration:

    A5_GLOBAL_SEED=2026
    A5_STAGE_GATE_RESIDUAL=0.25
    A5_TIG_DILATED_OFFSETS="-16,-8,-4,-2,0,2,4,8,16"

Raw StageFusion result:

| Metric | Mean ± Std |
|---|---:|
| Acc | 76.8866 ± 1.1008 |
| Edit | 76.8037 ± 1.8513 |
| F1@0.10 | 65.4468 ± 8.7340 |
| F1@0.25 | 61.6212 ± 8.3820 |
| F1@0.50 | 51.7880 ± 6.5420 |

## 5. Visual-only expert

Main script:

    scripts/sics155/run_sics155_100_visualonly_trainval_multisplit_safe.sh

Original source:

    m10/a5_tig_stagefusion_r025_rsens_seeded/run_sics155_100_visualonly_trainval_multisplit_safe.sh

Usage pattern:

    cd /path/to/MS-TCN2
    SPLITS="1 2 3 4" EPOCHS="100" bash scripts/sics155/run_sics155_100_visualonly_trainval_multisplit_safe.sh

The visual-only expert disables structured evidence groups using A5_ABLATE switches.

Raw visual-only result:

| Metric | Mean ± Std |
|---|---:|
| Acc | 75.5443 ± 2.3995 |
| Edit | 73.9611 ± 2.8771 |
| F1@0.10 | 67.5828 ± 3.5817 |
| F1@0.25 | 63.5800 ± 3.5573 |
| F1@0.50 | 53.4634 ± 1.9759 |

## 6. One-frame de-spiking

Tool:

    tools/smooth_eval_dataset_v2.py

Run StageFusion smoothing:

    python tools/smooth_eval_dataset_v2.py --dataset sics155_100_a5_tig_stagefusion_r025_rsens_seed2026_trainval --split 1 --min_len 2 --suffix smooth_v2_2

Run visual-only smoothing:

    python tools/smooth_eval_dataset_v2.py --dataset sics155_100_a5_visualonly_seed2026_trainval --split 1 --min_len 2 --suffix smooth_v2_2

Recommended loop:

    for sp in 1 2 3 4; do
      python tools/smooth_eval_dataset_v2.py --dataset sics155_100_a5_tig_stagefusion_r025_rsens_seed2026_trainval --split ${sp} --min_len 2 --suffix smooth_v2_2
      python tools/smooth_eval_dataset_v2.py --dataset sics155_100_a5_visualonly_seed2026_trainval --split ${sp} --min_len 2 --suffix smooth_v2_2
    done

StageFusion + de-spiking result:

| Metric | Mean ± Std |
|---|---:|
| Acc | 76.8894 ± 1.1027 |
| Edit | 87.0774 ± 1.0886 |
| F1@0.10 | 81.0845 ± 7.5273 |
| F1@0.25 | 76.3728 ± 7.0275 |
| F1@0.50 | 64.2230 ± 5.9979 |

Visual-only + de-spiking result:

| Metric | Mean ± Std |
|---|---:|
| Acc | 75.5426 ± 2.3973 |
| Edit | 85.4306 ± 3.4039 |
| F1@0.10 | 82.2565 ± 2.8783 |
| F1@0.25 | 77.4284 ± 3.9625 |
| F1@0.50 | 65.1871 ± 2.6416 |

## 7. Reliability-aware fallback

Tool:

    tools/make_sics_fallback_eval.py

The fallback uses visual-only predictions only when input-side structured evidence quality is low:

    eye_valid_mean < 0.30 or mask_shape_valid_mean < 0.10

Run:

    for sp in 1 2 3 4; do
      python tools/make_sics_fallback_eval.py --split ${sp} --out_dataset sics155_100_a5_stagefusion_smooth2_vofallback_quality_trainval --use_frag 0
    done

Final result:

| Metric | Mean ± Std |
|---|---:|
| Acc | 77.3081 ± 1.1754 |
| Edit | 87.9522 ± 1.0700 |
| F1@0.10 | 86.0560 ± 1.2154 |
| F1@0.25 | 81.2107 ± 1.3959 |
| F1@0.50 | 68.4079 ± 1.9326 |

## 8. Collect four-fold metrics

Tool:

    tools/collect_sics_4fold_metrics.py

Example:

    python tools/collect_sics_4fold_metrics.py --dataset sics155_100_a5_stagefusion_smooth2_vofallback_quality_trainval

## 9. Notes

The SICS155_100 validation demonstrates that GeoIG-StageFusion remains effective under more complex surgical workflows and larger structured-evidence quality variation.

The reliability-aware fallback acts as a conservative safeguard for rare degraded-evidence cases rather than replacing the StageFusion pathway.

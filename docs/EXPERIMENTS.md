> Historical GeoIG release documentation. Retained for provenance and existing utilities; see README.md, docs/CANONICAL_IMPLEMENTATION.md and docs/RELEASE_SCOPE.md for the current AERI source release. Historical result tables and fallback settings are not the current paper's result/protocol declaration.

# Experiment Summary

This document summarizes the main experimental results included in the release.

## 1. Metrics

The reported metrics are:

| Metric | Meaning |
|---|---|
| Acc | Frame-wise accuracy. |
| Edit | Segment-level edit score. |
| F1@0.10 | Segmental F1 at 10 percent overlap threshold. |
| F1@0.25 | Segmental F1 at 25 percent overlap threshold. |
| F1@0.50 | Segmental F1 at 50 percent overlap threshold. |

All results are reported as mean ± standard deviation across four folds unless otherwise specified.

## 2. Cataract101 main result

GeoIG-StageFusion-r0.25:

| Metric | Mean ± Std |
|---|---:|
| Acc | 91.4460 ± 4.3981 |
| Edit | 87.1281 ± 7.8541 |
| F1@0.10 | 90.2695 ± 6.6507 |
| F1@0.25 | 88.9021 ± 8.0818 |
| F1@0.50 | 83.3764 ± 11.2133 |

Cataract101 is the main dataset for method validation, component analysis, and mechanism visualization.

## 3. SICS155_100 robust validation

### 3.1 StageFusion raw

| Split | Acc | Edit | F1@0.10 | F1@0.25 | F1@0.50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 76.4538 | 76.5801 | 52.5568 | 49.5739 | 41.7614 |
| 2 | 76.7688 | 74.0443 | 70.3633 | 65.3920 | 54.3021 |
| 3 | 75.6628 | 79.1887 | 75.8763 | 72.3711 | 59.7938 |
| 4 | 78.6609 | 77.4015 | 62.9908 | 59.1479 | 51.2949 |
| Mean ± Std | 76.8866 ± 1.1008 | 76.8037 ± 1.8513 | 65.4468 ± 8.7340 | 61.6212 ± 8.3820 | 51.7880 ± 6.5420 |

### 3.2 StageFusion + one-frame de-spiking

| Split | Acc | Edit | F1@0.10 | F1@0.25 | F1@0.50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 76.4591 | 85.7423 | 68.0773 | 64.2134 | 54.0938 |
| 2 | 76.7718 | 86.7644 | 85.9813 | 79.9065 | 66.5888 |
| 3 | 75.6608 | 87.0349 | 84.5977 | 80.6897 | 66.4368 |
| 4 | 78.6657 | 88.7680 | 85.6818 | 80.6818 | 69.7727 |
| Mean ± Std | 76.8894 ± 1.1027 | 87.0774 ± 1.0886 | 81.0845 ± 7.5273 | 76.3728 ± 7.0275 | 64.2230 ± 5.9979 |

### 3.3 Visual-only raw

| Split | Acc | Edit | F1@0.10 | F1@0.25 | F1@0.50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 73.1313 | 76.5488 | 67.3413 | 63.8454 | 52.6219 |
| 2 | 73.2562 | 69.9361 | 65.2712 | 59.5623 | 51.5699 |
| 3 | 77.2160 | 76.8222 | 73.4774 | 69.1552 | 56.7780 |
| 4 | 78.5737 | 72.5373 | 64.2413 | 61.7569 | 52.8838 |
| Mean ± Std | 75.5443 ± 2.3995 | 73.9611 ± 2.8771 | 67.5828 ± 3.5817 | 63.5800 ± 3.5573 | 53.4634 ± 1.9759 |

### 3.4 Visual-only + one-frame de-spiking

| Split | Acc | Edit | F1@0.10 | F1@0.25 | F1@0.50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 73.1316 | 87.3993 | 84.2348 | 79.8619 | 66.0529 |
| 2 | 73.2562 | 79.7379 | 77.3393 | 70.5750 | 61.1048 |
| 3 | 77.2150 | 86.0417 | 84.3292 | 79.3687 | 65.1635 |
| 4 | 78.5676 | 88.5437 | 83.1228 | 79.9082 | 68.4271 |
| Mean ± Std | 75.5426 ± 2.3973 | 85.4306 ± 3.4039 | 82.2565 ± 2.8783 | 77.4284 ± 3.9625 | 65.1871 ± 2.6416 |

### 3.5 StageFusion + de-spiking + reliability-aware fallback

| Split | Acc | Edit | F1@0.10 | F1@0.25 | F1@0.50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 78.1340 | 89.2414 | 87.9630 | 83.5648 | 70.8333 |
| 2 | 76.7718 | 86.7644 | 85.9813 | 79.9065 | 66.5888 |
| 3 | 75.6608 | 87.0349 | 84.5977 | 80.6897 | 66.4368 |
| 4 | 78.6657 | 88.7680 | 85.6818 | 80.6818 | 69.7727 |
| Mean ± Std | 77.3081 ± 1.1754 | 87.9522 ± 1.0700 | 86.0560 ± 1.2154 | 81.2107 ± 1.3959 | 68.4079 ± 1.9326 |

## 4. Interpretation

The SICS155_100 results show three important effects.

First, one-frame de-spiking substantially improves Edit and segmental F1 while preserving frame-wise accuracy. This indicates that a major error source is short prediction fragmentation rather than global phase collapse.

Second, the visual-only expert is more stable under severely degraded structured evidence, but its frame-wise accuracy is lower than StageFusion in general.

Third, the reliability-aware fallback combines the strengths of StageFusion and visual-only inference, improving both mean performance and fold-wise stability.

## 5. Paper usage

Recommended table usage:

| Table | Content |
|---|---|
| Main Cataract101 table | GeoIG-StageFusion-r0.25 main comparison. |
| Cataract101 component table | Checkpoint-controlled component masking. |
| SICS155_100 main table | Comparison against SICS baselines and final method. |
| SICS robust inference table | StageFusion raw, StageFusion smooth, visual-only smooth, and fallback. |

SICS155_100 should be described as additional validation and robustness analysis, not as a replacement for the main Cataract101 experiments.

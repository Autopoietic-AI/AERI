# Canonical Implementation

This document clarifies the exact implementation used for the released GeoIG-StageFusion experiments.

## Final canonical source

The released canonical implementation is:

```text
src/geoig_stagefusion/main_cat101_a5_tig_stagefusion.py
src/geoig_stagefusion/model_a5_tig_stagefusion.py
These files are copied from:

m10/a5_tig_stagefusion_r025_rsens_seeded/main_cat101_a5_tig_stagefusion.py
m10/a5_tig_stagefusion_r025_rsens_seeded/model_a5_tig_stagefusion.py

The model file is identical to the original Cat101 StageFusion implementation, while the seeded main entry adds deterministic seed control:

A5_GLOBAL_SEED=2026
Final model

The final released model is GeoIG-StageFusion-r0.25.

Main configuration:

A5_STAGE_GATE_RESIDUAL=0.25
A5_TIG_DILATED_OFFSETS="-16,-8,-4,-2,0,2,4,8,16"
A5_GLOBAL_SEED=2026
Ablation and visual-only support

The released model supports:

A5_ABLATE_GEO
A5_ABLATE_TRAJ
A5_ABLATE_MASKDYN
A5_ABLATE_REL
A5_ABLATE_AUX
A5_ABLATE_DISTSPEED
A5_ABLATE_TRAJ_DIMS

Therefore, the same canonical model file supports:

GeoIG-StageFusion expert
Visual-only expert
Component masking / ablation analysis
Cataract101

Main four-fold script:

scripts/cataract101/run_cat101_a5_tig_stagefusion_r025_multisplit.sh
SICS155_100

The final SICS155_100 reported method uses:

StageFusion expert
+ one-frame de-spiking
+ reliability-aware fallback

StageFusion expert:

scripts/sics155/run_sics155_100_rsens_seed2026_r025_trainval_multisplit.sh

Visual-only expert:

scripts/sics155/run_sics155_100_visualonly_trainval_multisplit_safe.sh

One-frame de-spiking:

tools/smooth_eval_dataset_v2.py

Reliability-aware fallback:

tools/make_sics_fallback_eval.py

Fallback trigger:

eye_valid_mean < 0.30 or mask_shape_valid_mean < 0.10


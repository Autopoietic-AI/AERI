# Canonical GeoIG-StageFusion Contract

## Architecture
- visual feature: 384D
- W-TIG offsets: -16,-8,-4,-2,0,2,4,8,16
- RC-StageFusion residual: r=0.25
- MS-TCN++: 4 stages, 10 layers, 64 feature maps
- main_dim: 256
- guide_dim: 64
- beta_max: 0.20
- optimizer: Adam
- learning rate: 5e-4
- epochs: 100
- smooth_w: 0.15
- gate_l1: 0.001

## Canonical visual encoder
M8-v1.2-derived A3b-v12 dual-gate encoder.

Checkpoint:
`weights/visual_encoder/cat101_a3b_v12_dualgate_split1_best.pth`

The same checkpoint generated the current canonical 384D features for both
Cataract101 and SICS-155.

## Final structured input contract
For every video:
- `<id>.npy` visual feature: shape `(384,T)`
- EASE `.npz` containing:
  `geo`, `aux`, `presence_mask`, `cls_proxy`, `cls_proxy_mask`,
  `traj`, `eye_center`, `eye_radius`, `inst_tips`
- mask-shape `.npz` containing:
  `mask_shape`, `mask_shape_valid`
- frame-level ground truth sequence
- mapping and video-level split bundles

## Seed-42 controlled retraining
- visual_only: `A5_VISUAL_ONLY=1`
- local_structured: `A5_ABLATE_TIG=1`, `r=0`
- wtig_fixed: W-TIG enabled, `r=0`
- full: W-TIG enabled, `r=0.25`

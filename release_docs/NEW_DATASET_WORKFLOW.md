# New Dataset Workflow

The model side is frozen. A new dataset needs only a dataset adapter.

## Required dataset-normalization layer

Create:
- `mapping.txt`
- `groundTruth/<video>.txt`
- `splits/train.splitK.bundle`
- `splits/test.splitK.bundle`
- optionally `val.splitK.bundle`

Then run/prepare:
1. YOLO eye + instrument assets
2. SAM2 packed masks
3. A3b-v12 384D features
4. EASE assets
5. mask-shape assets

Once all three final inputs exist:
- `features/*.npy`
- `geo_assets/*.npz`
- `maskshape/*.npz`

the StageFusion model itself is dataset-agnostic and the fixed Seed-42
four-way ablation launcher can be used.

The exact raw-video filename parser / phase-annotation parser cannot be frozen
until the new dataset is present. Put that small dataset-specific adapter under:
`new_dataset_template/adapters/<dataset_name>/`.

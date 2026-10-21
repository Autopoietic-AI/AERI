# AERI

**AERI: Reliability-Aware Instrument–Eye Interaction Reasoning for Cataract Surgical Phase Recognition**

AERI (Anatomy-Anchored Evidence Reasoning and Integration) combines a mask-guided visual anchor with EASE structured evidence, W-TIG temporal reasoning, reliability-constrained branch fusion, and an MS-TCN++ phase decoder.

This is a **code-only source release**. Model weights, trained checkpoints, raw videos, generated masks/features and prediction caches are not distributed. The project is published under its current paper name, AERI; historical GeoIG Python filenames and class names are retained for compatibility.

## Source layout

| Path | Contents |
| --- | --- |
| `stagefusion/` | Full core model, ablation model, training/prediction entry, evaluation, and the MS-TCN++ dependency. |
| `cat101_native/` | Cataract101 visual-model training, feature extraction, detection, SAM2 masks, EASE and mask-shape builders. |
| `sics_cross/` | SICS detection, segmentation, feature extraction and structured-asset preparation. |
| `cataract1k_contract/` | Supplied Cataract-1K class mapping, split files and input metadata. |
| `release_docs/` and `reports/` | Original snapshot documentation and provenance reports; historical paths are preserved. |
| `aeri.py` | Convenient entry to the original temporal training and prediction program. |
| `preprocessing/`, `scripts/`, `src/geoig_stagefusion/` | Existing release utilities and compatibility paths, retained from the former repository. |

The full implementations are directly available in:

- `stagefusion/model_a5_tig_stagefusion.py`
- `stagefusion/model_a5_tig_stagefusion_ablation.py`
- `cat101_native/7_train_m8_igcoattn_v1_2_dualgate_framecls.py`

The twenty Python files supplied in the native snapshot are included without changing their computational implementation. See [source manifest](docs/source_snapshot_manifest.json).

## Install and inspect

```bash
python -m pip install -r requirements.txt
python tools/check_release.py
python aeri.py --help
```

Install [SAM2](https://github.com/facebookresearch/sam2) separately when generating masks. PyTorch/torchvision should be installed for the intended CPU/CUDA environment. No pretrained checkpoint is bundled or downloaded by the release checker.

## Training and prediction

Prepare the dataset's frame labels, class mapping and split bundles under `DATA_ROOT/data/DATASET/`. Supply the visual features, structured assets and mask-shape assets explicitly:

```bash
python aeri.py --action train --root /path/to/project --dataset DATASET \
  --m8_feature_dir /path/to/features --m7_asset_dir /path/to/structured_assets \
  --maskshape_dir /path/to/maskshape/npz --split 1

python aeri.py --action predict --root /path/to/project --dataset DATASET \
  --m8_feature_dir /path/to/features --m7_asset_dir /path/to/structured_assets \
  --maskshape_dir /path/to/maskshape/npz --split 1
```

These are path templates. Use the relevant dataset configuration and original command-line options; historical defaults point to the authors' server. The main entry supports `train` and `predict`; its input feature dimension defaults to 384. See [current implementation notes](docs/CANONICAL_IMPLEMENTATION.md) and [release scope](docs/RELEASE_SCOPE.md).

## Weights and reproducibility scope

**No weights are included**, including the two detector checkpoints present in the original local snapshot. Supply checkpoints externally for detection, segmentation and visual-feature extraction. A source release without those assets does not provide immediate inference with the paper's fitted models. Dataset access conditions and the required upstream training assets still apply.

Historical GeoIG result tables and experimental notes are archived under their existing documentation paths. They are not the current AERI paper's result statement. This update publishes the supplied source and repairs missing release dependencies; it does not retrain models or claim to reproduce every reported number.

## Attribution and licensing

The bundled MS-TCN++ implementation is from the [official MS-TCN2 repository](https://github.com/sj-li/MS-TCN2); its pinned revision and MIT notice are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Other upstream projects retain their respective licenses. No new project-wide license is assigned to the authors' original code by this upload.

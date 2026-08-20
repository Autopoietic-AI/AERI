# Release Checklist

This checklist should be completed before pushing the repository to GitHub.

## 1. Code organization

- [ ] Core model files are under src/geoig_stagefusion.
- [ ] Preprocessing files are grouped by function.
- [ ] Cataract101 scripts are under scripts/cataract101.
- [ ] SICS155_100 scripts are under scripts/sics155.
- [ ] Evaluation and diagnostic tools are under tools.
- [ ] Documentation is under docs.

## 2. Large files

Confirm that the repository does not contain:

- [ ] Raw videos.
- [ ] Extracted frames.
- [ ] .npy visual feature files.
- [ ] .npz YOLO / SAM2 / GeoIG / maskshape assets.
- [ ] Model checkpoints.
- [ ] Prediction files.
- [ ] Training logs.
- [ ] Diagnostic outputs.
- [ ] Result folders.

## 3. Documentation

Confirm that the following files exist:

- [ ] README.md
- [ ] docs/CANONICAL_IMPLEMENTATION.md
- [ ] docs/FILE_MANIFEST.md
- [ ] docs/PREPROCESSING.md
- [ ] docs/PIPELINE_CATARACT101.md
- [ ] docs/PIPELINE_SICS155.md
- [ ] docs/EXPERIMENTS.md
- [ ] docs/DATASETS.md
- [ ] docs/RELEASE_CHECKLIST.md

## 4. Canonical implementation

Confirm that the released canonical model is:

    src/geoig_stagefusion/model_a5_tig_stagefusion.py
    src/geoig_stagefusion/main_cat101_a5_tig_stagefusion.py

Confirm that these files were copied from:

    m10/a5_tig_stagefusion_r025_rsens_seeded/

Confirm that the model supports:

    A5_ABLATE_GEO
    A5_ABLATE_TRAJ
    A5_ABLATE_MASKDYN
    A5_ABLATE_REL
    A5_ABLATE_AUX
    A5_ABLATE_DISTSPEED
    A5_ABLATE_TRAJ_DIMS

## 5. Main configuration

Confirm that the final configuration is documented:

    A5_GLOBAL_SEED=2026
    A5_STAGE_GATE_RESIDUAL=0.25
    A5_TIG_DILATED_OFFSETS="-16,-8,-4,-2,0,2,4,8,16"

## 6. Reproducibility

- [ ] Scripts include or document the expected ROOT path.
- [ ] Scripts include or document the expected DATA_ROOT path.
- [ ] Dataset layouts are documented.
- [ ] Preprocessing pipeline is documented.
- [ ] Main experiment commands are documented.
- [ ] Expected metrics are documented.
- [ ] Environment requirements are documented.

## 7. Path cleanup

Before release, search for machine-specific paths:

    /root/autodl-tmp
    autodl
    MS-TCN2

These should either be removed or clearly documented as examples.

## 8. Final result records

Confirm that the following final results are documented:

- [ ] Cataract101 GeoIG-StageFusion-r0.25.
- [ ] SICS155_100 StageFusion raw.
- [ ] SICS155_100 StageFusion + de-spiking.
- [ ] SICS155_100 visual-only raw.
- [ ] SICS155_100 visual-only + de-spiking.
- [ ] SICS155_100 final reliability-aware fallback.

## 9. License and citation

Before public release:

- [ ] Add LICENSE.
- [ ] Add citation information.
- [ ] Add BibTeX once the paper title and author list are finalized.
- [ ] Check dataset license compatibility.
- [ ] Check pretrained model license compatibility.


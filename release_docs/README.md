# GeoIG-StageFusion Release v1

This is the frozen working release for the current JBHI manuscript.

Read in order:
1. `docs/CANONICAL_CONTRACT.md`
2. `docs/PIPELINE.md`
3. `docs/KNOWN_AUDIT_NOTES.md`
4. `docs/NEW_DATASET_WORKFLOW.md`

Main runtime:
- final full code: `code/stagefusion/full_seeded/`
- ablation code: `code/stagefusion/ablation/`
- preprocessing: `code/preprocess/`
- A3b-v12 visual encoder: `code/visual_encoder/a3b_v12/`
- EASE builders: `code/ease/`
- evaluation: `code/evaluation/`
- historical Seed42 reproductions: `reproduction/`

For a new dataset:
`launchers/create_new_dataset_workspace.sh <name>`

Once standardized assets are ready:
`launchers/run_4way_ablation_seed42.sh <name>`

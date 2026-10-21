# AERI implementation map

The current release follows the native source snapshot supplied by the authors. These are the implementation files, not placeholders:

| Component | Source |
| --- | --- |
| Visual encoder and frame-level training | `cat101_native/7_train_m8_igcoattn_v1_2_dualgate_framecls.py` and `cat101_native/train_cat101_a3b_v12_dualgate_framecls.py` |
| Visual feature extraction | `cat101_native/extract_cat101_a3b_v12_dualgate_features.py`, `sics_cross/extract_sics155_100_a3b_v12_dualgate_features.py` |
| EASE construction | `cat101_native/build_cat101_a4_geo_assets_*.py`, corresponding `sics_cross/` builders |
| Local/temporal evidence and branch fusion | `stagefusion/model_a5_tig_stagefusion.py` |
| Ablation implementation | `stagefusion/model_a5_tig_stagefusion_ablation.py` |
| Temporal training/prediction | `stagefusion/main_cat101_a5_tig_stagefusion.py`, or `aeri.py` |
| Decoder dependency | `stagefusion/model.py` (attributed upstream MS-TCN++) |
| Metrics | `stagefusion/eval.py` |

Internal GeoIG/A5 names and the supplied source filenames are retained to keep imports and state-dictionary conventions stable. AERI is the paper and repository name.

The actual training entry reads `A5_GLOBAL_SEED` (default 2026), model options and the dataset arguments from the supplied source. Read the provided scripts/configuration for each experiment; source defaults are not a substitute for identifying the experiment being run.

The previously published `src/geoig_stagefusion/` paths now contain compatibility entries. Full model code lives in `stagefusion/`.

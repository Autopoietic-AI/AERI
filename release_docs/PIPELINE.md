# Full Pipeline

1. Raw surgical video + frame-level phase labels
2. Eye / instrument YOLO
3. SAM2 mask refinement
4. A3b-v12 visual encoder → 384D feature sequence
5. EASE asset construction
   - eye anchor
   - instrument tip / center-proxy
   - trajectory
   - geometry
   - auxiliary cues
   - source/reliability cues
6. Mask-shape asset construction
7. W-TIG sparse temporal evidence reasoning
8. RC-StageFusion conservative fusion (`r=0.25`)
9. MS-TCN++ phase decoding
10. Acc / Edit / F1@0.10 / F1@0.25 / F1@0.50
11. SICS-only practical configuration may additionally apply the frozen
    reliability-aware safeguard protocol; v3 is the canonical corrected parser.

# qualitative_robustness_appendix — notes

Figure: 3 rows (fog, frost, snow) x 4 columns (clean left | corrupted left | clean-only baseline (E0) |Δd| | Full |Δd|).
Drift map = |d_corrupted − d_clean| of the *same* model (corruption-induced prediction change, NOT a GT error).
No training, no inference: everything is read from existing files. Regenerate with
`CUDA_VISIBLE_DEVICES="" python3 scripts/figures/qualitative_robustness_appendix.py figure --vmax 50`
(candidate list + contact sheets: `... candidates`).

## Selected examples (all from the 28-frame primary subset, robust20 severity 3)

| row | scene:frame | E0 mean |Δd| | Full mean |Δd| | E0 sky |Δd| | Full sky |Δd| | sky fraction | rank by E0 drift (frames with sky ≥ 5 %) |
|---|---|---|---|---|---|---|---|
| fog | 0038:0072 | 55.87 | 13.38 | 107.94 | 25.56 | 0.51 | 3 |
| frost | 0045:0066 | 19.35 | 1.92 | 41.41 | 3.63 | 0.46 | 4 |
| snow | 0047:0222 | 15.19 | 0.28 | 32.76 | 0.08 | 0.29 | 2 |

Why:
- **fog 0038:0072** and **frost 0045:0066** are the examples chosen by the automatic rule of
  `scripts/final_analysis.py` (frames with sky ≥ 5 %, top-5 by E0 drift, the one with the MEDIAN reduction ratio), i.e.
  typical rather than best cases.
  Fog shows the tile-shaped failure in uniformly fogged sky and the residual of the full model (25.6 px in sky).
  Frost shows drift in the sky AND on the textured mountains/background where the frost pattern creates spurious
  matches; Full removes most of it (a residual sky block remains).
- **snow 0047:0222** replaces the automatic pick (0045:0164) ONLY for scene diversity: the automatic snow pick is the
  same scene and almost the same view as the frost row. 0047:0222 is rank 2 of 28 by E0 snow drift; its reduction
  (−98 %) is higher than the dataset-level snow reduction (−91 % overall, −93 % in sky), so this row is on the favourable
  side of typical. The rank-1 snow frame
  (0045:0197) was not used for the same scene-duplication reason. The worst fog frame (0038:0001, E0 141.7 → Full 44.1 px)
  was not used because it is an outlier; it is listed in `qualitative/candidates_ranked.csv`.
- Dataset-level numbers that the figure illustrates (`regional_summary.json`): sky |Δd| E0 → Full: fog 50.9 → 12.0,
  frost 21.4 → 2.2, snow 11.2 → 0.8 px (mean over the 28 frames).

Ranked candidate lists (top-6 per corruption) and contact sheets (`candidates_{fog,frost,snow}.png`) are produced by the
`candidates` command of the figure script.

## Source files
- Predictions (cached by `scripts/final_analysis.py predict`, tiled inference 352x704 / overlap 0.7, fp32):
  `outputs/track2/final_analysis/cache/{E0,Full}_<scene>_<frame>_{clean,fog,frost,snow}.npz` (key `pred`)
  - E0 = `checkpoints/track2/croco_clean_teacher.pt`
  - Full = `outputs/track2/track2_croco_x7_x6b_mono/stageC_seed2026/final.pt` (seed 2026; the local matched model, not the
    train-all submission checkpoint)
- Corrupted inputs: `outputs/track2/robust20_val/<cond>/<scene>/frame_left_<frame>.png` (frozen cache, severity 3)
- Clean inputs / GT for the sky mask: Spring validation split via `track2.spring_data.read_frame`
- Per-frame numbers: `outputs/track2/final_analysis/regional_stats.csv`, `qualitative/candidates_ranked.csv`
- Sky = finite GT disparity == 0 (GT used only for this mask and the inset numbers, never shown).

## Colour scale
One colormap (magma) and ONE shared range 0–50 px for all six drift maps; values above 50 px are clipped (colourbar
arrow). Clipped pixels: fog E0 13.1 % / Full 6.6 %, frost E0 13.3 % / Full 0.0 %, snow E0 11.8 % / Full 0.0 %.
50 px was chosen so that the frost/snow rows stay visible next to fog (fogged-sky drift of E0 exceeds 100 px).
Inset means are computed on the unclipped maps. Images are shown at half resolution (960x540) per panel.

## Caveats
- Local synthetic corruptions (ImageNet-C style re-implementation + depth-aware fog), not RobustSpring test images.
- One training seed of Full (2026); seed spread of fog drift is large (4.64 ± 1.79 px over 3 seeds).
- Drift is not error: a small drift does not by itself mean a correct disparity.

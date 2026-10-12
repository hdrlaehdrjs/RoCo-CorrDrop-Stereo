# Regional drift statistics for all three seeds (local development results, 28 primary frames)

Inference only. For each seed the clean-only baseline and the full model trained with that seed are compared (descriptive pairing: the two recipes differ in more than one factor). Drift = |d_corr − d_clean| in px, mean over the 28 frames of the per-frame regional mean; sky = finite ground-truth disparity 0; textured = 15×15 grey local std ≥ 2 on the corrupted left image.

Sky pixel fraction (ground truth, identical for all seeds): 29.1 %

| quantity | seed 2026 | seed 2027 | seed 2028 | mean ± sd |
|---|---|---|---|---|
| fog: share of baseline drift located in sky (%) | 58.6 | 57.5 | 52.6 | 56.3 ± 3.2 |
| fog: sky drift, clean-only baseline | 50.91 | 43.57 | 30.83 | 41.77 ± 10.16 |
| fog: sky drift, Full | 11.98 | 18.74 | 8.36 | 13.03 ± 5.27 |
| fog: sky drift reduction (%) | 76 | 57 | 73 | 69 ± 10 |
| fog: textured-region drift, clean-only baseline | 0.206 | 0.206 | 0.210 | 0.207 ± 0.002 |
| fog: textured-region drift, Full | 0.212 | 0.218 | 0.218 | 0.216 ± 0.003 |
| fog: image-wide drift, clean-only baseline | 16.80 | 14.66 | 10.49 | 13.98 ± 3.21 |
| fog: image-wide drift, Full | 4.29 | 6.58 | 3.06 | 4.64 ± 1.79 |
| frost: share of baseline drift located in sky (%) | 50.5 | 46.0 | 43.3 | 46.6 ± 3.6 |
| frost: sky drift, clean-only baseline | 21.37 | 6.68 | 5.40 | 11.15 ± 8.87 |
| frost: sky drift, Full | 2.23 | 1.49 | 1.17 | 1.63 ± 0.55 |
| frost: sky drift reduction (%) | 90 | 78 | 78 | 82 ± 7 |
| frost: textured-region drift, clean-only baseline | 8.838 | 3.655 | 2.759 | 5.084 ± 3.282 |
| frost: textured-region drift, Full | 1.115 | 0.830 | 0.755 | 0.900 ± 0.190 |
| frost: image-wide drift, clean-only baseline | 9.63 | 3.92 | 2.94 | 5.50 ± 3.61 |
| frost: image-wide drift, Full | 1.15 | 0.86 | 0.76 | 0.92 ± 0.20 |
| snow: share of baseline drift located in sky (%) | 50.5 | 46.2 | 43.2 | 46.6 ± 3.7 |
| snow: sky drift, clean-only baseline | 11.20 | 3.69 | 2.43 | 5.78 ± 4.74 |
| snow: sky drift, Full | 0.80 | 1.11 | 0.71 | 0.87 ± 0.21 |
| snow: sky drift reduction (%) | 93 | 70 | 71 | 78 ± 13 |
| snow: textured-region drift, clean-only baseline | 5.813 | 2.053 | 1.735 | 3.200 ± 2.268 |
| snow: textured-region drift, Full | 0.548 | 0.703 | 0.518 | 0.590 ± 0.099 |
| snow: image-wide drift, clean-only baseline | 5.80 | 2.05 | 1.73 | 3.20 ± 2.26 |
| snow: image-wide drift, Full | 0.55 | 0.70 | 0.52 | 0.59 ± 0.10 |

## Frame-level accounting

| quantity | seed 2026 | seed 2027 | seed 2028 |
|---|---|---|---|
| fog: frames (of 28) on which Full drifts more than the baseline | 8 | 8 | 10 |
| fog: largest such regression (px) | 0.016 | 0.002 | 0.005 |
| fog: largest sky fraction among those frames (%) | 4.5 | 4.5 | 4.5 |
| fog: worst baseline frame → Full on the same frame (px) | 0038:0001: 141.7 → 44.1 | 0038:0001: 125.1 → 59.9 | 0038:0001: 102.6 → 31.6 |
| fog: worst Full frame (px) | 0038:0001: 44.1 | 0038:0001: 59.9 | 0038:0001: 31.6 |
| frost: frames (of 28) on which Full drifts more than the baseline | 0 | 0 | 0 |
| frost: largest such regression (px) | 0.000 | 0.000 | 0.000 |
| frost: largest sky fraction among those frames (%) | 0.0 | 0.0 | 0.0 |
| frost: worst baseline frame → Full on the same frame (px) | 0039:0038: 62.2 → 5.2 | 0039:0038: 38.6 → 2.5 | 0039:0038: 22.5 → 3.6 |
| frost: worst Full frame (px) | 0038:0144: 5.8 | 0038:0144: 7.7 | 0038:0144: 4.4 |
| snow: frames (of 28) on which Full drifts more than the baseline | 0 | 1 | 0 |
| snow: largest such regression (px) | 0.000 | 4.658 | 0.000 |
| snow: largest sky fraction among those frames (%) | 0.0 | 71.6 | 0.0 |
| snow: worst baseline frame → Full on the same frame (px) | 0045:0197: 24.7 → 0.5 | 0039:0076: 8.8 → 0.1 | 0039:0076: 6.2 → 0.1 |
| snow: worst Full frame (px) | 0038:0144: 2.7 | 0038:0120: 6.3 | 0038:0144: 2.6 |

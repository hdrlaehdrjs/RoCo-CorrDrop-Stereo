# Same-checkpoint monocular ablation

Production `tiled_predict` (352×704 tiles, overlap 0.7, fp32) of each unchanged full-model checkpoint, once as is (fused) and once under `track2.mono.mono_disabled` (stereo-only: the stereo head output and its own confidences drive the tile blending). 28 primary frames × (clean + 20 corruptions); local development results.

| full-model checkpoint (seed) | Clean Abs fused / stereo-only | Clean D1 fused / stereo-only | ΔAbs20 fused / stereo-only | ΔAbs20 difference (fused − stereo-only) | mean / max pixelwise |fused − stereo-only| (px) | fused path vs regular evaluation (max |dev|) |
|---|---|---|---|---|---|---|
| 2026 | 0.2865 / 0.2866 | 0.5595 / 0.5594 | 0.6555 / 0.6556 | -0.00003 | 0.00102 / 1.287 | 0.00e+00 |
| 2027 | 0.2857 / 0.2856 | 0.5586 / 0.5586 | 0.7830 / 0.7830 | -0.00001 | 0.00242 / 1.003 | 0.00e+00 |
| 2028 | 0.2874 / 0.2873 | 0.5554 / 0.5552 | 0.5704 / 0.5703 | +0.00004 | 0.00387 / 4.414 | 0.00e+00 |

| seed | fog ΔAbs fused / stereo-only | frost | snow |
|---|---|---|---|
| 2026 | 4.2883 / 4.2888 | 1.1515 / 1.1513 | 0.5487 / 0.5486 |
| 2027 | 6.5797 / 6.5801 | 0.8606 / 0.8600 | 0.7044 / 0.7041 |
| 2028 | 3.0613 / 3.0681 | 0.7620 / 0.7611 | 0.5188 / 0.5179 |

## Fusion gate (stereo weight w_f; d = w_f·d_stereo + (1 − w_f)·d_mono), all 21 conditions pooled

| seed | min w_f | fraction of pixels with w_f < 0.99 | < 0.95 | < 0.90 | median w_f under fog |
|---|---|---|---|---|---|
| 2026 | 0.98793 | 0.000161 | 0 | 0 | 0.9997 |
| 2027 | 0.98363 | 0.0021 | 0 | 0 | 0.9991 |
| 2028 | 0.98795 | 2.84e-07 | 0 | 0 | 0.9995 |

## Error against ground truth under fog (px): monocular-only vs stereo-only prediction

| seed | sky: mono / stereo | textureless: mono / stereo |
|---|---|---|
| 2026 | 4.1 / 11.8 | 4.4 / 6.6 |
| 2027 | 14.0 / 18.6 | 10.3 / 10.4 |
| 2028 | 7.4 / 8.3 | 6.5 / 4.8 |

Per-condition gate statistics, regional errors of the fused / stereo-only / monocular-only predictions and the fractions of pixels whose error improves or worsens: `mono_same_checkpoint.csv` (one row per checkpoint and condition).

LaTeX rows of Table 9:

```latex
2026 & 0.2865 / 0.2866 & 0.6555 / 0.6556 & 0.988 & 0.016\% \\
2027 & 0.2857 / 0.2856 & 0.7830 / 0.7830 & 0.984 & 0.21\% \\
2028 & 0.2874 / 0.2873 & 0.5704 / 0.5703 & 0.988 & $<$0.001\% \\
```

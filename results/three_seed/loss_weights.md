# Loss-weight sensitivity over three seeds

Local development results: 28 primary frames (scenes 0038/0039/0045/0047) of the scene-disjoint 49-frame validation split, frozen robust20 cache, final checkpoint. ΔAbs20 = unweighted mean over the 20 corruptions of the mean |d_corr − d_clean|. Not official RobustSpring results. Seeds 2026 / 2027 / 2028; sd = sample standard deviation (ddof = 1) across independently trained runs, not a standard error or a confidence interval; three runs do not support significance claims.

Objective (`scripts/train_track2.py`): `L = λ_clean·L_clean + λ_anchor·L_anchor + λ_mono·L_mono(clean) + η(t)·[λ_corr·L_corr + λ_Δ·L_Δ] + λ_mono·L_mono(corrupted)`, η(t) = min(1, t/(0.1·T)). Default (λ_clean, λ_anchor, λ_corr, λ_Δ, λ_mono) = (1, 0.1, 0.25, 1, 0.5). One coefficient differs per row; `- L_delta` is the λ_Δ = 0 ablation and Full is the λ_Δ = 1.0 / λ_corr = 0.25 row.

## Consistency weight λ_Δ (0 / 0.5 / 1.0 / 2.0)

| variant | ΔAbs20 (2026 / 2027 / 2028) | mean ± sd | matched-seed Δ vs Full (2026 / 2027 / 2028) | mean ± sd of Δ | seeds worse than Full | Clean Abs | Clean D1 | Fog | Frost | Snow | Motion blur |
|---|---|---|---|---|---|---|---|---|---|---|---|
| - L_delta | 0.949 / 0.981 / 0.842 | 0.924 ± 0.073 | +0.294 / +0.198 / +0.272 | +0.255 ± 0.050 | 3 of 3 | 0.2814 ± 0.0024 | 0.564 ± 0.002 | 8.26 ± 1.14 | 2.00 ± 0.63 | 0.63 ± 0.18 | 0.78 ± 0.05 |
| lambda_delta = 0.5 | 0.790 / 0.808 / 0.704 | 0.767 ± 0.056 | +0.134 / +0.025 / +0.133 | +0.097 ± 0.063 | 3 of 3 | 0.2846 ± 0.0038 | 0.555 ± 0.002 | 6.35 ± 0.86 | 1.07 ± 0.19 | 0.57 ± 0.08 | 0.70 ± 0.01 |
| **Full model** | 0.656 / 0.783 / 0.570 | 0.670 ± 0.107 | – | – | – | 0.2865 ± 0.0008 | 0.558 ± 0.002 | 4.64 ± 1.79 | 0.92 ± 0.20 | 0.59 ± 0.10 | 0.69 ± 0.00 |
| lambda_delta = 2.0 | 0.841 / 0.718 / 1.369 | 0.976 ± 0.346 | +0.186 / -0.065 / +0.798 | +0.306 ± 0.444 | 2 of 3 | 0.4289 ± 0.2357 | 1.406 ± 1.457 | 5.36 ± 1.51 | 2.83 ± 2.52 | 1.05 ± 0.31 | 1.17 ± 0.65 |

## Corrupted-supervision weight λ_corr (0.125 / 0.25 / 0.5)

| variant | ΔAbs20 (2026 / 2027 / 2028) | mean ± sd | matched-seed Δ vs Full (2026 / 2027 / 2028) | mean ± sd of Δ | seeds worse than Full | Clean Abs | Clean D1 | Fog | Frost | Snow | Motion blur |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lambda_corr = 0.125 | 0.790 / 0.791 / 0.751 | 0.778 ± 0.023 | +0.135 / +0.008 / +0.181 | +0.108 ± 0.089 | 3 of 3 | 0.2879 ± 0.0026 | 0.557 ± 0.004 | 6.19 ± 0.19 | 1.05 ± 0.39 | 0.68 ± 0.02 | 0.72 ± 0.03 |
| **Full model** | 0.656 / 0.783 / 0.570 | 0.670 ± 0.107 | – | – | – | 0.2865 ± 0.0008 | 0.558 ± 0.002 | 4.64 ± 1.79 | 0.92 ± 0.20 | 0.59 ± 0.10 | 0.69 ± 0.00 |
| lambda_corr = 0.5 | 0.786 / 0.767 / 0.708 | 0.754 ± 0.041 | +0.130 / -0.016 / +0.137 | +0.084 ± 0.086 | 2 of 3 | 0.2874 ± 0.0022 | 0.558 ± 0.001 | 5.74 ± 0.87 | 0.99 ± 0.09 | 0.69 ± 0.08 | 0.73 ± 0.03 |

### Clean accuracy, matched seeds (variant − Full)

| variant | Clean Abs Δ in px (2026 / 2027 / 2028) | mean ± sd | Clean D1 Δ in points (2026 / 2027 / 2028) |
|---|---|---|---|
| - L_delta | -0.0078 / -0.0037 / -0.0039 | -0.0051 ± 0.0023 | +0.006 / +0.004 / +0.008 |
| lambda_delta = 0.5 | -0.0040 / -0.0035 / +0.0016 | -0.0020 ± 0.0031 | -0.002 / -0.004 / -0.002 |
| lambda_delta = 2.0 | +0.0054 / +0.0080 / +0.4136 | +0.1423 ± 0.2349 | -0.000 / +0.012 / +2.532 |
| lambda_corr = 0.125 | +0.0003 / +0.0004 / +0.0036 | +0.0014 ± 0.0019 | -0.006 / -0.003 / +0.006 |
| lambda_corr = 0.5 | -0.0011 / +0.0012 / +0.0025 | +0.0008 ± 0.0018 | -0.002 / +0.001 / +0.003 |

### Per-corruption ΔAbs (mean over the verified seeds; n in the header)

| corruption | - L_delta (n=3) | lambda_delta = 0.5 (n=3) | Full model (n=3) | lambda_delta = 2.0 (n=3) | lambda_corr = 0.125 (n=3) | lambda_corr = 0.5 (n=3) |
|---|---|---|---|---|---|---|
| brightness | 0.051 | 0.050 | 0.051 | 0.087 | 0.051 | 0.051 |
| contrast | 0.061 | 0.062 | 0.064 | 0.097 | 0.063 | 0.064 |
| defocus_blur | 0.251 | 0.254 | 0.257 | 0.547 | 0.256 | 0.254 |
| elastic_transform | 1.051 | 1.090 | 1.168 | 1.040 | 1.124 | 1.130 |
| fog | 8.261 | 6.354 | 4.643 | 5.364 | 6.186 | 5.741 |
| frost | 2.003 | 1.066 | 0.925 | 2.828 | 1.049 | 0.995 |
| gaussian_blur | 0.213 | 0.215 | 0.217 | 0.487 | 0.217 | 0.213 |
| gaussian_noise | 0.580 | 0.574 | 0.494 | 0.772 | 0.598 | 0.553 |
| glass_blur | 0.328 | 0.329 | 0.333 | 0.614 | 0.330 | 0.330 |
| impulse_noise | 0.556 | 0.507 | 0.462 | 0.762 | 0.492 | 0.529 |
| jpeg_compression | 0.326 | 0.303 | 0.289 | 0.398 | 0.291 | 0.293 |
| motion_blur | 0.777 | 0.700 | 0.691 | 1.168 | 0.721 | 0.733 |
| pixelate | 0.140 | 0.138 | 0.136 | 0.313 | 0.135 | 0.136 |
| rain | 0.112 | 0.109 | 0.109 | 0.140 | 0.111 | 0.109 |
| saturate | 0.055 | 0.054 | 0.055 | 0.073 | 0.054 | 0.055 |
| shot_noise | 0.525 | 0.471 | 0.425 | 0.703 | 0.463 | 0.474 |
| snow | 0.634 | 0.567 | 0.591 | 1.047 | 0.680 | 0.689 |
| spatter | 0.476 | 0.452 | 0.443 | 0.684 | 0.689 | 0.620 |
| speckle_noise | 0.472 | 0.435 | 0.418 | 0.643 | 0.442 | 0.488 |
| zoom_blur | 1.614 | 1.610 | 1.623 | 1.750 | 1.600 | 1.619 |

Plots: `loss_lambda_delta.png`, `loss_lambda_corr.png`.

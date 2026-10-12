# CorrMask sampling probability over three seeds

Local development results: 28 primary frames (scenes 0038/0039/0045/0047) of the scene-disjoint 49-frame validation split, frozen robust20 cache, final checkpoint. ΔAbs20 = unweighted mean over the 20 corruptions of the mean |d_corr − d_clean|. Not official RobustSpring results. Seeds 2026 / 2027 / 2028; sd = sample standard deviation (ddof = 1) across independently trained runs, not a standard error or a confidence interval; three runs do not support significance claims.

`s2aug.p_mask` is the CorrMask probability inside the S2Aug gate (p_aug = 0.7 after the 10 % curriculum): nominal unconditional frequency 0.7·p_mask. The full model is the p_mask = 0.25 row. Structured correspondence dropout (decoder cross-attention, p = 0.5) is unchanged in every row.

## Three-seed summary

| variant | ΔAbs20 (2026 / 2027 / 2028) | mean ± sd | matched-seed Δ vs Full (2026 / 2027 / 2028) | mean ± sd of Δ | seeds worse than Full | Clean Abs | Clean D1 | Fog | Frost | Snow | Motion blur |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CorrMask p_mask = 0 | 0.853 / 0.615 / 0.807 | 0.758 ± 0.126 | +0.197 / -0.168 / +0.236 | +0.088 ± 0.223 | 2 of 3 | 0.2880 ± 0.0030 | 0.559 ± 0.006 | 5.72 ± 2.00 | 1.10 ± 0.22 | 0.72 ± 0.15 | 0.73 ± 0.02 |
| CorrMask p_mask = 0.10 | 0.946 / 0.778 / 0.703 | 0.809 ± 0.125 | +0.291 / -0.005 / +0.132 | +0.139 ± 0.148 | 2 of 3 | 0.2866 ± 0.0042 | 0.556 ± 0.004 | 6.15 ± 1.38 | 1.42 ± 0.38 | 0.85 ± 0.36 | 0.76 ± 0.06 |
| **Full model** | 0.656 / 0.783 / 0.570 | 0.670 ± 0.107 | – | – | – | 0.2865 ± 0.0008 | 0.558 ± 0.002 | 4.64 ± 1.79 | 0.92 ± 0.20 | 0.59 ± 0.10 | 0.69 ± 0.00 |
| CorrMask p_mask = 0.50 | 0.633 / 0.724 / 0.735 | 0.698 ± 0.056 | -0.022 / -0.058 / +0.165 | +0.028 ± 0.120 | 1 of 3 | 0.2861 ± 0.0063 | 0.555 ± 0.002 | 5.17 ± 0.60 | 0.97 ± 0.21 | 0.57 ± 0.08 | 0.70 ± 0.01 |

### Observed CorrMask event frequency (from each run's training log)

| setting | seed | events / all training samples | events / augmented samples | S2Aug fraction |
|---|---|---|---|---|
| CorrMask p_mask = 0 | 2026 | 0.0000 | 0.0000 | 0.6825 |
| CorrMask p_mask = 0 | 2027 | 0.0000 | 0.0000 | 0.6823 |
| CorrMask p_mask = 0 | 2028 | 0.0000 | 0.0000 | 0.6831 |
| CorrMask p_mask = 0.10 | 2026 | 0.0709 | 0.1039 | 0.6825 |
| CorrMask p_mask = 0.10 | 2027 | 0.0696 | 0.1020 | 0.6823 |
| CorrMask p_mask = 0.10 | 2028 | 0.0699 | 0.1023 | 0.6831 |
| Full (p_mask = 0.25) | 2026 | 0.1706 | 0.2500 | 0.6825 |
| Full (p_mask = 0.25) | 2027 | 0.1711 | 0.2507 | 0.6823 |
| Full (p_mask = 0.25) | 2028 | 0.1721 | 0.2519 | 0.6831 |
| CorrMask p_mask = 0.50 | 2026 | 0.3401 | 0.4982 | 0.6825 |
| CorrMask p_mask = 0.50 | 2027 | 0.3406 | 0.4991 | 0.6823 |
| CorrMask p_mask = 0.50 | 2028 | 0.3414 | 0.4999 | 0.6831 |

### Clean accuracy, matched seeds (variant − Full)

| variant | Clean Abs Δ in px (2026 / 2027 / 2028) | mean ± sd | Clean D1 Δ in points (2026 / 2027 / 2028) |
|---|---|---|---|
| CorrMask p_mask = 0 | +0.0021 / -0.0010 / +0.0033 | +0.0015 ± 0.0022 | +0.004 / -0.006 / +0.005 |
| CorrMask p_mask = 0.10 | +0.0005 / -0.0035 / +0.0032 | +0.0001 ± 0.0034 | -0.000 / -0.007 / +0.001 |
| CorrMask p_mask = 0.50 | -0.0029 / -0.0043 / +0.0058 | -0.0005 ± 0.0055 | -0.007 / -0.003 / +0.001 |

### Per-corruption ΔAbs (mean over the verified seeds; n in the header)

| corruption | CorrMask p_mask = 0 (n=3) | CorrMask p_mask = 0.10 (n=3) | Full model (n=3) | CorrMask p_mask = 0.50 (n=3) |
|---|---|---|---|---|
| brightness | 0.050 | 0.051 | 0.051 | 0.051 |
| contrast | 0.065 | 0.065 | 0.064 | 0.064 |
| defocus_blur | 0.256 | 0.253 | 0.257 | 0.251 |
| elastic_transform | 1.231 | 1.102 | 1.168 | 1.145 |
| fog | 5.721 | 6.153 | 4.643 | 5.168 |
| frost | 1.099 | 1.416 | 0.925 | 0.973 |
| gaussian_blur | 0.217 | 0.213 | 0.217 | 0.210 |
| gaussian_noise | 0.535 | 0.589 | 0.494 | 0.502 |
| glass_blur | 0.328 | 0.326 | 0.333 | 0.328 |
| impulse_noise | 0.488 | 0.520 | 0.462 | 0.463 |
| jpeg_compression | 0.309 | 0.302 | 0.289 | 0.302 |
| motion_blur | 0.730 | 0.758 | 0.691 | 0.696 |
| pixelate | 0.135 | 0.135 | 0.136 | 0.136 |
| rain | 0.110 | 0.108 | 0.109 | 0.109 |
| saturate | 0.054 | 0.055 | 0.055 | 0.055 |
| shot_noise | 0.438 | 0.474 | 0.425 | 0.437 |
| snow | 0.723 | 0.848 | 0.591 | 0.572 |
| spatter | 0.660 | 0.737 | 0.443 | 0.490 |
| speckle_noise | 0.406 | 0.453 | 0.418 | 0.387 |
| zoom_blur | 1.605 | 1.617 | 1.623 | 1.612 |

Plot: `corrmask.png`.

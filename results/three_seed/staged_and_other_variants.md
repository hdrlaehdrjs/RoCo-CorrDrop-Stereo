# Staged variants and additional variants over three seeds

Local development results: 28 primary frames (scenes 0038/0039/0045/0047) of the scene-disjoint 49-frame validation split, frozen robust20 cache, final checkpoint. ΔAbs20 = unweighted mean over the 20 corruptions of the mean |d_corr − d_clean|. Not official RobustSpring results. Seeds 2026 / 2027 / 2028; sd = sample standard deviation (ddof = 1) across independently trained runs, not a standard error or a confidence interval; three runs do not support significance claims.

These rows are separate recipes, not one-factor ablations of the full model: the clean-only baseline uses no teacher, the short fine-tuning starts from the seed-2026 clean-only checkpoint and trains decoder + head for 1,351 steps, the consistency stage, reliability gating and the attention-supervision variants have no monocular branch. Every robust variant uses the seed-2026 clean-only checkpoint as its frozen teacher for every seed. `- monocular branch` is the staged row `+ corr. dropout + overlays`.

## Three-seed summary

| variant | ΔAbs20 (2026 / 2027 / 2028) | mean ± sd | matched-seed Δ vs Full (2026 / 2027 / 2028) | mean ± sd of Δ | seeds worse than Full | Clean Abs | Clean D1 | Fog | Frost | Snow | Motion blur |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clean-only baseline | 2.195 / 1.531 / 1.247 | 1.658 ± 0.486 | (different recipe) | – | – | 0.2842 ± 0.0024 | 0.567 ± 0.002 | 13.98 ± 3.21 | 5.50 ± 3.61 | 3.20 ± 2.26 | 1.56 ± 0.44 |
| Clean-corrupted consistency | 1.487 / 1.508 / 0.736 | 1.244 ± 0.440 | (different recipe) | – | – | 0.2828 ± 0.0046 | 0.554 ± 0.003 | 9.94 ± 4.06 | 5.16 ± 3.18 | 1.32 ± 0.68 | 1.71 ± 0.54 |
| - monocular branch (= '+ corr. dropout + overlays') | 0.766 / 0.688 / 0.627 | 0.694 ± 0.070 | +0.111 / -0.095 / +0.056 | +0.024 ± 0.106 | 2 of 3 | 0.2860 ± 0.0036 | 0.555 ± 0.003 | 5.00 ± 1.07 | 1.03 ± 0.29 | 0.63 ± 0.12 | 0.70 ± 0.03 |
| **Full model** | 0.656 / 0.783 / 0.570 | 0.670 ± 0.107 | – | – | – | 0.2865 ± 0.0008 | 0.558 ± 0.002 | 4.64 ± 1.79 | 0.92 ± 0.20 | 0.59 ± 0.10 | 0.69 ± 0.00 |
| Short robust fine-tuning | 1.678 / 1.515 / 1.868 | 1.687 ± 0.177 | (different recipe) | – | – | 0.2817 ± 0.0010 | 0.556 ± 0.001 | 14.62 ± 1.73 | 6.70 ± 1.22 | 2.93 ± 0.49 | 1.46 ± 0.17 |
| Reliability gating | 1.074 / 0.786 / 0.800 | 0.887 ± 0.163 | (different recipe) | – | – | 0.2862 ± 0.0047 | 0.559 ± 0.002 | 7.65 ± 1.86 | 1.57 ± 0.85 | 0.89 ± 0.14 | 0.76 ± 0.08 |
| Attention supervision | 1.275 / 1.376 / 1.426 | 1.359 ± 0.077 | (different recipe) | – | – | 0.2900 ± 0.0045 | 0.589 ± 0.006 | 15.01 ± 0.82 | 3.40 ± 0.30 | 0.74 ± 0.13 | 1.65 ± 0.52 |
| Attention supervision + corr. dropout | 0.927 / 0.998 / 0.973 | 0.966 ± 0.036 | (different recipe) | – | – | 0.2886 ± 0.0051 | 0.603 ± 0.018 | 7.11 ± 1.10 | 2.65 ± 0.75 | 0.91 ± 0.08 | 1.21 ± 0.20 |

### All 49 validation frames (secondary)

| variant | Clean Abs (mean) | ΔAbs20 (2026 / 2027 / 2028) | mean |
|---|---|---|---|
| Clean-only baseline | 10.198 | 15.025 / 11.621 / 10.129 | 12.258 |
| Clean-corrupted consistency | 10.146 | 9.302 / 9.416 / 6.364 | 8.361 |
| - monocular branch (= '+ corr. dropout + overlays') | 10.492 | 6.999 / 6.204 / 6.739 | 6.647 |
| Full model | 10.433 | 6.188 / 7.056 / 6.102 | 6.449 |

### Reliability gating: logged gate value (mean over the last quarter of training)

| seed | clean branch | corrupted branch |
|---|---|---|
| 2026 | 0.9953 | 0.9953 |
| 2027 | 0.9955 | 0.9955 |
| 2028 | 0.9956 | 0.9956 |

### Per-corruption ΔAbs (mean over the verified seeds; n in the header)

| corruption | Clean-only baseline (n=3) | Clean-corrupted consistency (n=3) | - monocular branch (= '+ corr. dropout + overlays') (n=3) | Full model (n=3) | Short robust fine-tuning (n=3) | Reliability gating (n=3) | Attention supervision (n=3) | Attention supervision + corr. dropout (n=3) |
|---|---|---|---|---|---|---|---|---|
| brightness | 0.051 | 0.050 | 0.050 | 0.051 | 0.049 | 0.050 | 0.053 | 0.054 |
| contrast | 0.063 | 0.061 | 0.064 | 0.064 | 0.060 | 0.065 | 0.064 | 0.063 |
| defocus_blur | 0.260 | 0.249 | 0.254 | 0.257 | 0.260 | 0.254 | 0.237 | 0.234 |
| elastic_transform | 1.124 | 1.083 | 1.150 | 1.168 | 1.138 | 1.094 | 0.909 | 0.955 |
| fog | 13.984 | 9.936 | 5.000 | 4.643 | 14.618 | 7.646 | 15.014 | 7.107 |
| frost | 5.496 | 5.162 | 1.033 | 0.925 | 6.695 | 1.569 | 3.399 | 2.654 |
| gaussian_blur | 0.226 | 0.210 | 0.214 | 0.217 | 0.226 | 0.215 | 0.195 | 0.192 |
| gaussian_noise | 0.894 | 0.490 | 0.489 | 0.494 | 0.676 | 0.567 | 0.500 | 0.721 |
| glass_blur | 0.339 | 0.319 | 0.329 | 0.333 | 0.322 | 0.330 | 0.309 | 0.311 |
| impulse_noise | 0.878 | 0.444 | 0.486 | 0.462 | 0.688 | 0.529 | 0.435 | 0.643 |
| jpeg_compression | 0.379 | 0.298 | 0.297 | 0.289 | 0.370 | 0.289 | 0.271 | 0.297 |
| motion_blur | 1.563 | 1.707 | 0.697 | 0.691 | 1.456 | 0.761 | 1.655 | 1.207 |
| pixelate | 0.143 | 0.135 | 0.135 | 0.136 | 0.141 | 0.134 | 0.122 | 0.120 |
| rain | 0.186 | 0.118 | 0.107 | 0.109 | 0.158 | 0.109 | 0.126 | 0.115 |
| saturate | 0.055 | 0.054 | 0.054 | 0.055 | 0.054 | 0.054 | 0.056 | 0.057 |
| shot_noise | 0.862 | 0.405 | 0.428 | 0.425 | 0.652 | 0.453 | 0.413 | 0.595 |
| snow | 3.196 | 1.315 | 0.627 | 0.591 | 2.925 | 0.889 | 0.743 | 0.912 |
| spatter | 1.117 | 0.856 | 0.449 | 0.443 | 1.116 | 0.680 | 0.645 | 0.739 |
| speckle_noise | 0.740 | 0.381 | 0.410 | 0.418 | 0.549 | 0.436 | 0.410 | 0.686 |
| zoom_blur | 1.606 | 1.605 | 1.603 | 1.623 | 1.588 | 1.609 | 1.615 | 1.658 |

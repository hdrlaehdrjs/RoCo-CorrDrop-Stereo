# Core ablations over three seeds

Local development results: 28 primary frames (scenes 0038/0039/0045/0047) of the scene-disjoint 49-frame validation split, frozen robust20 cache, final checkpoint. ΔAbs20 = unweighted mean over the 20 corruptions of the mean |d_corr − d_clean|. Not official RobustSpring results. Seeds 2026 / 2027 / 2028; sd = sample standard deviation (ddof = 1) across independently trained runs, not a standard error or a confidence interval; three runs do not support significance claims.

## Three-seed summary

| variant | ΔAbs20 (2026 / 2027 / 2028) | mean ± sd | matched-seed Δ vs Full (2026 / 2027 / 2028) | mean ± sd of Δ | seeds worse than Full | Clean Abs | Clean D1 | Fog | Frost | Snow | Motion blur |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Full model** | 0.656 / 0.783 / 0.570 | 0.670 ± 0.107 | – | – | – | 0.2865 ± 0.0008 | 0.558 ± 0.002 | 4.64 ± 1.79 | 0.92 ± 0.20 | 0.59 ± 0.10 | 0.69 ± 0.00 |
| - correspondence dropout | 0.894 / 0.839 / 0.946 | 0.893 ± 0.054 | +0.239 / +0.056 / +0.375 | +0.223 ± 0.160 | 3 of 3 | 0.2886 ± 0.0051 | 0.556 ± 0.002 | 7.78 ± 1.08 | 1.82 ± 0.53 | 0.70 ± 0.16 | 1.02 ± 0.16 |
| Random-key dropout | 1.385 / 1.061 / 1.087 | 1.177 ± 0.180 | +0.729 / +0.278 / +0.516 | +0.508 ± 0.226 | 3 of 3 | 0.2871 ± 0.0056 | 0.561 ± 0.010 | 11.11 ± 1.77 | 3.42 ± 0.83 | 0.97 ± 0.61 | 1.30 ± 0.45 |
| - L_delta | 0.949 / 0.981 / 0.842 | 0.924 ± 0.073 | +0.294 / +0.198 / +0.272 | +0.255 ± 0.050 | 3 of 3 | 0.2814 ± 0.0024 | 0.564 ± 0.002 | 8.26 ± 1.14 | 2.00 ± 0.63 | 0.63 ± 0.18 | 0.78 ± 0.05 |
| Image-plane mask transport | 0.858 / 0.880 / 0.742 | 0.827 ± 0.074 | +0.202 / +0.097 / +0.172 | +0.157 ± 0.054 | 3 of 3 | 0.2910 ± 0.0034 | 0.554 ± 0.003 | 6.29 ± 0.73 | 1.39 ± 0.47 | 0.85 ± 0.25 | 0.80 ± 0.09 |
| - teacher / anchor | 0.921 / 0.738 / 1.363 | 1.007 ± 0.321 | +0.265 / -0.045 / +0.793 | +0.338 ± 0.424 | 2 of 3 | 0.3065 ± 0.0091 | 0.627 ± 0.041 | 6.16 ± 1.80 | 3.53 ± 1.85 | 1.21 ± 0.47 | 1.42 ± 1.01 |
| - monocular branch (= '+ corr. dropout + overlays') | 0.766 / 0.688 / 0.627 | 0.694 ± 0.070 | +0.111 / -0.095 / +0.056 | +0.024 ± 0.106 | 2 of 3 | 0.2860 ± 0.0036 | 0.555 ± 0.003 | 5.00 ± 1.07 | 1.03 ± 0.29 | 0.63 ± 0.12 | 0.70 ± 0.03 |
| - sensor overlays | 0.774 / 0.658 / 0.867 | 0.766 ± 0.105 | +0.118 / -0.125 / +0.297 | +0.097 ± 0.212 | 2 of 3 | 0.2882 ± 0.0043 | 0.558 ± 0.002 | 5.72 ± 1.25 | 1.05 ± 0.54 | 0.80 ± 0.27 | 0.80 ± 0.16 |

### Clean accuracy, matched seeds (variant − Full)

| variant | Clean Abs Δ in px (2026 / 2027 / 2028) | mean ± sd | Clean D1 Δ in points (2026 / 2027 / 2028) |
|---|---|---|---|
| - correspondence dropout | +0.0031 / -0.0025 / +0.0058 | +0.0021 ± 0.0042 | -0.004 / -0.004 / +0.003 |
| Random-key dropout | +0.0035 / -0.0050 / +0.0032 | +0.0006 ± 0.0048 | +0.011 / -0.007 / +0.005 |
| - L_delta | -0.0078 / -0.0037 / -0.0039 | -0.0051 ± 0.0023 | +0.006 / +0.004 / +0.008 |
| Image-plane mask transport | +0.0046 / +0.0018 / +0.0069 | +0.0044 ± 0.0026 | -0.009 / -0.001 / -0.001 |
| - teacher / anchor | +0.0305 / +0.0158 / +0.0136 | +0.0199 ± 0.0092 | +0.115 / +0.041 / +0.052 |
| - monocular branch (= '+ corr. dropout + overlays') | +0.0005 / -0.0038 / +0.0016 | -0.0006 ± 0.0028 | -0.006 / -0.007 / +0.003 |
| - sensor overlays | +0.0067 / -0.0004 / -0.0013 | +0.0017 ± 0.0043 | -0.002 / -0.001 / +0.005 |

### Random-key control: masking budget (from each run's training log)

Per selected query, the control masks as many right-view keys as the structured epipolar mask would, drawn uniformly without replacement.

| seed | steps with dropout | keys of the structured mask | keys masked by the control | of them inside the epipolar band | share of the 132-key band budget |
|---|---|---|---|---|---|
| 2026 | 5324 | 131.3 | 131.3 | 17.83 | 13.5 % |
| 2027 | 5323 | 131.2 | 131.2 | 17.83 | 13.5 % |
| 2028 | 5335 | 131.2 | 131.2 | 17.83 | 13.5 % |

Expected under uniform sampling: 132 / 968 = 13.6 %.

### Per-corruption ΔAbs (mean over the verified seeds; n in the header)

| corruption | Full model (n=3) | - correspondence dropout (n=3) | Random-key dropout (n=3) | - L_delta (n=3) | Image-plane mask transport (n=3) | - teacher / anchor (n=3) | - monocular branch (= '+ corr. dropout + overlays') (n=3) | - sensor overlays (n=3) |
|---|---|---|---|---|---|---|---|---|
| brightness | 0.051 | 0.050 | 0.050 | 0.051 | 0.051 | 0.051 | 0.050 | 0.050 |
| contrast | 0.064 | 0.063 | 0.063 | 0.061 | 0.066 | 0.067 | 0.064 | 0.066 |
| defocus_blur | 0.257 | 0.253 | 0.253 | 0.251 | 0.255 | 0.261 | 0.254 | 0.256 |
| elastic_transform | 1.168 | 1.104 | 1.094 | 1.051 | 1.176 | 1.336 | 1.150 | 1.128 |
| fog | 4.643 | 7.777 | 11.110 | 8.261 | 6.291 | 6.156 | 5.000 | 5.722 |
| frost | 0.925 | 1.820 | 3.422 | 2.003 | 1.388 | 3.529 | 1.033 | 1.048 |
| gaussian_blur | 0.217 | 0.213 | 0.214 | 0.213 | 0.215 | 0.216 | 0.214 | 0.217 |
| gaussian_noise | 0.494 | 0.524 | 0.479 | 0.580 | 0.604 | 0.615 | 0.489 | 0.590 |
| glass_blur | 0.333 | 0.327 | 0.322 | 0.328 | 0.326 | 0.321 | 0.329 | 0.333 |
| impulse_noise | 0.462 | 0.515 | 0.470 | 0.556 | 0.528 | 0.604 | 0.486 | 0.476 |
| jpeg_compression | 0.289 | 0.287 | 0.307 | 0.326 | 0.302 | 0.439 | 0.297 | 0.293 |
| motion_blur | 0.691 | 1.019 | 1.305 | 0.777 | 0.798 | 1.420 | 0.697 | 0.803 |
| pixelate | 0.136 | 0.135 | 0.135 | 0.140 | 0.138 | 0.135 | 0.135 | 0.135 |
| rain | 0.109 | 0.110 | 0.109 | 0.112 | 0.109 | 0.146 | 0.107 | 0.116 |
| saturate | 0.055 | 0.054 | 0.055 | 0.055 | 0.056 | 0.056 | 0.054 | 0.054 |
| shot_noise | 0.425 | 0.435 | 0.428 | 0.525 | 0.491 | 0.588 | 0.428 | 0.426 |
| snow | 0.591 | 0.697 | 0.972 | 0.634 | 0.850 | 1.206 | 0.627 | 0.796 |
| spatter | 0.443 | 0.465 | 0.758 | 0.476 | 0.743 | 0.810 | 0.449 | 0.763 |
| speckle_noise | 0.418 | 0.412 | 0.393 | 0.472 | 0.498 | 0.532 | 0.410 | 0.432 |
| zoom_blur | 1.623 | 1.596 | 1.611 | 1.614 | 1.648 | 1.655 | 1.603 | 1.620 |

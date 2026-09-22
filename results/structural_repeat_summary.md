# Structural repeats: image-plane transport and no teacher / anchor

28 primary validation frames (scenes 0038/0039/0045/0047), frozen robust20 severity-3 cache, final checkpoint of
each run, unchanged metric code. Sample sd (n-1). n = 2-3 seeds: descriptive only, **no significance claim**.

| Variant | Seeds | individual dAbs20 | mean +- sd | Clean Abs | Clean D1 | Fog | Frost | Snow | Motion blur |
|---|---:|---|---|---|---|---|---|---|---|
| Full | 3 | 0.656 / 0.783 / 0.570 | 0.670 +- 0.107 | 0.287 +- 0.001 | 0.558 +- 0.002 | 4.64 +- 1.79 | 0.92 +- 0.20 | 0.59 +- 0.10 | 0.69 +- 0.00 |
| image-plane transport | 2 | 0.858 / 0.880 | 0.869 +- 0.016 | 0.289 +- 0.003 | 0.554 +- 0.005 | 6.64 +- 0.57 | 1.53 +- 0.55 | 0.97 +- 0.18 | 0.85 +- 0.01 |
| no teacher / anchor | 2 | 0.921 / 0.738 | 0.829 +- 0.129 | 0.309 +- 0.011 | 0.637 +- 0.053 | 5.17 +- 0.80 | 2.50 +- 0.70 | 1.03 +- 0.51 | 0.84 +- 0.05 |

## Matched-seed comparison: image-plane transport vs Full
| seed | metric | image-plane transport | Full (same seed) | difference |
|---|---|---|---|---|
| 2026 | dabs20 | 0.858 | 0.656 | +0.202 |
| 2026 | clean_abs | 0.291 | 0.287 | +0.005 |
| 2026 | fog | 6.240 | 4.288 | +1.952 |
| 2026 | frost | 1.925 | 1.151 | +0.774 |
| 2026 | snow | 1.096 | 0.549 | +0.547 |
| 2027 | dabs20 | 0.880 | 0.783 | +0.097 |
| 2027 | clean_abs | 0.288 | 0.286 | +0.002 |
| 2027 | fog | 7.046 | 6.580 | +0.466 |
| 2027 | frost | 1.143 | 0.861 | +0.283 |
| 2027 | snow | 0.848 | 0.704 | +0.143 |

## Matched-seed comparison: no teacher / anchor vs Full
| seed | metric | no teacher / anchor | Full (same seed) | difference |
|---|---|---|---|---|
| 2026 | dabs20 | 0.921 | 0.656 | +0.265 |
| 2026 | clean_abs | 0.317 | 0.287 | +0.030 |
| 2026 | fog | 5.731 | 4.288 | +1.443 |
| 2026 | frost | 2.994 | 1.151 | +1.842 |
| 2026 | snow | 1.393 | 0.549 | +0.844 |
| 2027 | dabs20 | 0.738 | 0.783 | -0.045 |
| 2027 | clean_abs | 0.301 | 0.286 | +0.016 |
| 2027 | fog | 4.603 | 6.580 | -1.976 |
| 2027 | frost | 2.003 | 0.861 | +1.143 |
| 2027 | snow | 0.668 | 0.704 | -0.037 |

## Missing / excluded runs
- none

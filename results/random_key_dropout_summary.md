# Random-key dropout control

28 primary validation frames (scenes 0038/0039/0045/0047), frozen robust20 severity-3 cache, final checkpoint of
each run, unchanged metric code. Sample sd (n-1). n = 2-3 seeds: descriptive only, **no significance claim**.

| Variant | Seeds | individual dAbs20 | mean +- sd | Clean Abs | Clean D1 | Fog | Frost | Snow | Motion blur |
|---|---:|---|---|---|---|---|---|---|---|
| Full | 3 | 0.656 / 0.783 / 0.570 | 0.670 +- 0.107 | 0.287 +- 0.001 | 0.558 +- 0.002 | 4.64 +- 1.79 | 0.92 +- 0.20 | 0.59 +- 0.10 | 0.69 +- 0.00 |
| no correspondence dropout | 2 | 0.894 / 0.839 | 0.866 +- 0.039 | 0.286 +- 0.005 | 0.555 +- 0.000 | 7.62 +- 1.48 | 1.77 +- 0.74 | 0.61 +- 0.04 | 1.00 +- 0.22 |
| random-key dropout (control) | 2 | 1.385 / 1.061 | 1.223 +- 0.229 | 0.285 +- 0.007 | 0.561 +- 0.014 | 11.89 +- 1.63 | 3.64 +- 1.04 | 1.10 +- 0.80 | 1.08 +- 0.33 |

## Differences of means (dAbs20)
- random-key mean - Full mean (3 seeds) = structured epipolar dropout: +0.553
- random-key mean - Full mean over the SAME seeds (2026/2027): +0.503
- random-key mean - no-dropout mean: +0.356
- for scale: Full seed sd = 0.107

## Matched-seed comparison vs Full (structured dropout)
| seed | metric | random-key dropout (control) | Full (same seed) | difference |
|---|---|---|---|---|
| 2026 | dabs20 | 1.385 | 0.656 | +0.729 |
| 2026 | clean_abs | 0.290 | 0.287 | +0.004 |
| 2026 | fog | 13.041 | 4.288 | +8.753 |
| 2026 | frost | 4.377 | 1.151 | +3.225 |
| 2026 | snow | 1.666 | 0.549 | +1.117 |
| 2027 | dabs20 | 1.061 | 0.783 | +0.278 |
| 2027 | clean_abs | 0.281 | 0.286 | -0.005 |
| 2027 | fog | 10.735 | 6.580 | +4.155 |
| 2027 | frost | 2.905 | 0.861 | +2.044 |
| 2027 | snow | 0.538 | 0.704 | -0.167 |

## Matched-seed comparison vs no dropout
| seed | random-key | no dropout | difference |
|---|---|---|---|
| 2026 | 1.385 | 0.894 | +0.491 |
| 2027 | 1.061 | 0.839 | +0.222 |

## Missing / excluded runs
- none

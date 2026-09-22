# Seed ablation summary (28 normal-scene validation frames, robust20 severity-3 cache)

Mean +- sample standard deviation (n-1). n = 2-3 seeds: **no statistical significance is claimed**; the last
column expresses the difference to the Full mean in units of the Full seed sd (descriptive effect size only).

| variant | seeds | clean Abs | dAbs20 | fog | frost | snow | dAbs20 - Full | in Full sd |
|---|---|---|---|---|---|---|---|---|
| Full | 2026/2027/2028 | 0.287 +- 0.001 | 0.670 +- 0.107 | 4.64 +- 1.79 | 0.92 +- 0.20 | 0.59 +- 0.10 | - | - |
| no correspondence dropout | 2026/2027 | 0.286 +- 0.005 | 0.866 +- 0.039 | 7.62 +- 1.48 | 1.77 +- 0.74 | 0.61 +- 0.04 | +0.197 | +1.84 |
| no L_delta | 2026/2027 | 0.280 +- 0.002 | 0.965 +- 0.022 | 8.67 +- 1.25 | 2.29 +- 0.54 | 0.69 +- 0.21 | +0.296 | +2.76 |
| no mono head | 2026/2027 | 0.284 +- 0.004 | 0.727 +- 0.055 | 5.56 +- 0.63 | 1.12 +- 0.36 | 0.67 +- 0.13 | +0.058 | +0.54 |
| no sensor overlay | 2026/2027 | 0.289 +- 0.006 | 0.716 +- 0.082 | 5.25 +- 1.33 | 0.74 +- 0.11 | 0.66 +- 0.19 | +0.046 | +0.43 |

Per-seed dAbs20: Full: 2026=0.656, 2027=0.783, 2028=0.570; no correspondence dropout: 2026=0.894, 2027=0.839; no L_delta: 2026=0.949, 2027=0.981; no mono head: 2026=0.766, 2027=0.688; no sensor overlay: 2026=0.774, 2027=0.658

## Missing / excluded runs
- none

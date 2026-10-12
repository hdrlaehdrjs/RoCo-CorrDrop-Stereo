# Result files

Everything except the leaderboard numbers is a **local development result**: 4,054 training pairs, scene-disjoint
validation split, 28 primary frames (scenes 0038 / 0039 / 0045 / 0047), frozen robust20 cache, final checkpoint.
ΔAbs20 is the unweighted mean over the 20 corruptions of the mean |d_corr − d_clean|. Every local variant of the paper
has three independently trained runs (seeds 2026 / 2027 / 2028); "±" is the sample standard deviation across them.

## Where each table of the paper comes from

| paper | file |
|---|---|
| Table 1 (staged variants) | `three_seed/staged_and_other_variants.md`, `.csv` |
| Table 2, Table 6 (ablations, individual runs) | `three_seed/ablations.md`, `.csv` |
| Table 3, Table 14 (official results) | Spring / RobustSpring leaderboard; hashes of the uploaded files in `BUNDLE_CONTRACT.json`, `SHA256SUMS.txt` |
| Table 4 (hyperparameters) | `../configs/track2/`; resolved per run in `three_seed/config_check.md` |
| Table 5, Table 8, Table 13 (per-corruption values) | per-corruption section of `three_seed/staged_and_other_variants.md`; all runs in `three_seed/runs.csv` |
| Table 7 and the regional statements (sky share, textured regions, worst fog frame) | `three_seed/regional/regional_3seed.md`; per-frame rows in `three_seed/regional/rows_*.csv` |
| Figure 2 | `figures/` (seed-2026 checkpoints of the clean-only baseline and the full model) |
| Table 9 and the monocular-fusion diagnostic | `three_seed/mono/mono_same_checkpoint.md`, `.csv` |
| Inference overhead (Appendix A) | `three_seed/mono/inference_overhead.json` (idle GPU, 2026-10-08) |
| Table 10 (short fine-tuning, reliability gating, attention supervision) | `three_seed/staged_and_other_variants.md`, `.csv` |
| Table 11 (CorrMask probability) | `three_seed/corrmask.md`, `.csv`, `.png` |
| Table 12 (loss weights) | `three_seed/loss_weights.md`, `.csv`, `loss_lambda_*.png` |
| Table 15 (all 49 validation frames) | section "All 49 validation frames" of `three_seed/staged_and_other_variants.md` |
| Random-key masking budget (Appendix B) | section "Random-key control" of `three_seed/ablations.md` |
| LaTeX bodies of Tables 1, 2, 5–8, 10–13, 15 | `three_seed/latex_tables.tex` |

`three_seed/summary.md` lists all 21 variants in one table, `three_seed/runs.md` every one of the 63 runs with its
verification status, training time, peak memory and checkpoint hash, and `three_seed/manifest.json` the same in
machine-readable form. All files in `three_seed/` are written by the scripts in `../scripts/three_seed/`.

## Runs outside the statistics

- The seed-2028 run without teacher was repeated once with the same configuration and seed (ΔAbs20 1.002 instead of
  1.363). The tables use the first run; the repetition is listed in `three_seed/summary.md` and is not averaged in.
- Two runs (seed 2028 of the no-overlay ablation and of CorrMask p_mask = 0) stopped with a non-finite prediction
  and were repeated from scratch with the same configuration; only the repetitions are counted.
- One of the three λ_Δ = 2 runs became unstable after warm-up (clean Abs 0.701). It completed without a non-finite
  step and is counted.

## Earlier files

The files directly in this folder were written when the paper was submitted, when most ablations had two seeds:
`seed_ablation_summary.*`, `final_targeted_ablation_summary.md`, `random_key_dropout_summary.*`,
`structural_repeat_summary.*`, `per_corruption_targeted.csv`, `e0_baseline_summary.csv`. Their individual run values
are the seed-2026 / 2027 entries of `three_seed/`; their means are superseded by the three-seed means.
`regional_summary.json`, `fusion_weights.csv` and `mono_stereo_fused_error.csv` are the seed-2026 regional and
fusion analyses on four conditions (clean, fog, frost, snow); `inference_overhead.*` is the first latency measurement,
taken on a shared GPU.

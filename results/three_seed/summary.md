# All local variants — three-seed summary

Local development results: 28 primary frames (scenes 0038/0039/0045/0047) of the scene-disjoint 49-frame validation split, frozen robust20 cache, final checkpoint. ΔAbs20 = unweighted mean over the 20 corruptions of the mean |d_corr − d_clean|. Not official RobustSpring results. Seeds 2026 / 2027 / 2028; sd = sample standard deviation (ddof = 1) across independently trained runs, not a standard error or a confidence interval; three runs do not support significance claims.

| group | variant | verified seeds | ΔAbs20 (2026 / 2027 / 2028) | mean ± sd | Clean Abs mean ± sd |
|---|---|---|---|---|---|
| ablation | Full model | 3 / 3 | 0.656 / 0.783 / 0.570 | 0.670 ± 0.107 | 0.2865 ± 0.0008 |
| ablation | - correspondence dropout | 3 / 3 | 0.894 / 0.839 / 0.946 | 0.893 ± 0.054 | 0.2886 ± 0.0051 |
| ablation | Random-key dropout | 3 / 3 | 1.385 / 1.061 / 1.087 | 1.177 ± 0.180 | 0.2871 ± 0.0056 |
| ablation | - L_delta | 3 / 3 | 0.949 / 0.981 / 0.842 | 0.924 ± 0.073 | 0.2814 ± 0.0024 |
| ablation | Image-plane mask transport | 3 / 3 | 0.858 / 0.880 / 0.742 | 0.827 ± 0.074 | 0.2910 ± 0.0034 |
| ablation | - teacher / anchor | 3 / 3 | 0.921 / 0.738 / 1.363 | 1.007 ± 0.321 | 0.3065 ± 0.0091 |
| ablation | - monocular branch (= '+ corr. dropout + overlays') | 3 / 3 | 0.766 / 0.688 / 0.627 | 0.694 ± 0.070 | 0.2860 ± 0.0036 |
| ablation | - sensor overlays | 3 / 3 | 0.774 / 0.658 / 0.867 | 0.766 ± 0.105 | 0.2882 ± 0.0043 |
| corrmask | CorrMask p_mask = 0 | 3 / 3 | 0.853 / 0.615 / 0.807 | 0.758 ± 0.126 | 0.2880 ± 0.0030 |
| corrmask | CorrMask p_mask = 0.10 | 3 / 3 | 0.946 / 0.778 / 0.703 | 0.809 ± 0.125 | 0.2866 ± 0.0042 |
| corrmask | CorrMask p_mask = 0.50 | 3 / 3 | 0.633 / 0.724 / 0.735 | 0.698 ± 0.056 | 0.2861 ± 0.0063 |
| loss | lambda_delta = 0.5 | 3 / 3 | 0.790 / 0.808 / 0.704 | 0.767 ± 0.056 | 0.2846 ± 0.0038 |
| loss | lambda_delta = 2.0 | 3 / 3 | 0.841 / 0.718 / 1.369 | 0.976 ± 0.346 | 0.4289 ± 0.2357 |
| loss | lambda_corr = 0.125 | 3 / 3 | 0.790 / 0.791 / 0.751 | 0.778 ± 0.023 | 0.2879 ± 0.0026 |
| loss | lambda_corr = 0.5 | 3 / 3 | 0.786 / 0.767 / 0.708 | 0.754 ± 0.041 | 0.2874 ± 0.0022 |
| staged | Clean-only baseline | 3 / 3 | 2.195 / 1.531 / 1.247 | 1.658 ± 0.486 | 0.2842 ± 0.0024 |
| staged | Clean-corrupted consistency | 3 / 3 | 1.487 / 1.508 / 0.736 | 1.244 ± 0.440 | 0.2828 ± 0.0046 |
| staged | Short robust fine-tuning | 3 / 3 | 1.678 / 1.515 / 1.868 | 1.687 ± 0.177 | 0.2817 ± 0.0010 |
| staged | Reliability gating | 3 / 3 | 1.074 / 0.786 / 0.800 | 0.887 ± 0.163 | 0.2862 ± 0.0047 |
| staged | Attention supervision | 3 / 3 | 1.275 / 1.376 / 1.426 | 1.359 ± 0.077 | 0.2900 ± 0.0045 |
| staged | Attention supervision + corr. dropout | 3 / 3 | 0.927 / 0.998 / 0.973 | 0.966 ± 0.036 | 0.2886 ± 0.0051 |

## Runs that are not part of the statistics

- Repetition of `track2_croco_abl_no_teacher_s2028` (seed 2028; independent repetition of the seed-2028 run with the same configuration; reported next to it, not averaged): ΔAbs20 1.002, Clean Abs 0.2978, Clean D1 0.582. Counted run: ΔAbs20 1.363, Clean Abs 0.3010, Clean D1 0.608.
- `camera_ready_3seed_202610/stage1/runs/a7_no_overlay_s2028`: stopped with a non-finite prediction; kept unchanged and repeated from scratch with the same configuration (`camera_ready_3seed_202610/stage1/runs/a7_no_overlay_s2028_retry1`); only the repetition is counted.
- `camera_ready_3seed_202610/stage1/runs/b_cm_p000_s2028`: stopped with a non-finite prediction; kept unchanged and repeated from scratch with the same configuration (`camera_ready_3seed_202610/stage1/runs/b_cm_p000_s2028_retry1`); only the repetition is counted.

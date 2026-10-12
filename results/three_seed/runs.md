# Runs behind the three-seed tables

Local development results: 28 primary frames (scenes 0038/0039/0045/0047) of the scene-disjoint 49-frame validation split, frozen robust20 cache, final checkpoint. ΔAbs20 = unweighted mean over the 20 corruptions of the mean |d_corr − d_clean|. Not official RobustSpring results. Seeds 2026 / 2027 / 2028; sd = sample standard deviation (ddof = 1) across independently trained runs, not a standard error or a confidence interval; three runs do not support significance claims.

`verified`: training finished all steps with the registered seed on the 4,054-pair split, from the same initial / teacher checkpoints as the seed-2026 run of the variant; the recorded configuration equals the config file; no non-finite step; the evaluation has 1029 unique rows (588 on the primary frames) on the frozen robust20 cache and was run on the final checkpoint. Wall-clock hours are from a shared server (up to four concurrent runs) and are upper bounds.

| variant | seed | config | ΔAbs20 | Clean Abs | steps | train h | peak GiB | checkpoint sha256 | status |
|---|---|---|---|---|---|---|---|---|---|
| Full model | 2026 | `track2_croco_x7_x6b_mono` | 0.6555 | 0.2865 | 5405 | 9.6 | 17.0 | 0191e23da064 | verified |
| Full model | 2027 | `track2_croco_abl_seed2` | 0.7830 | 0.2857 | 5405 | 7.5 | 17.0 | a3f2689e51fe | verified |
| Full model | 2028 | `track2_croco_abl_seed3` | 0.5704 | 0.2874 | 5405 | 4.6 | 17.0 | 2d965c7898c0 | verified |
| - correspondence dropout | 2026 | `track2_croco_abl_no_dropout` | 0.8941 | 0.2896 | 5405 | 10.4 | 17.0 | b85f031b8b67 | verified |
| - correspondence dropout | 2027 | `track2_croco_abl_no_dropout_seed2` | 0.8385 | 0.2832 | 5405 | 5.0 | 17.0 | 5c70c6974e20 | verified |
| - correspondence dropout | 2028 | `track2_croco_abl_no_dropout_s2028` | 0.9458 | 0.2931 | 5405 | 6.1 | 17.0 | 2aec9af90bde | verified |
| Random-key dropout | 2026 | `track2_croco_abl_randkey_s1` | 1.3846 | 0.2900 | 5405 | 11.3 | 17.0 | aafb3354699d | verified |
| Random-key dropout | 2027 | `track2_croco_abl_randkey_s2` | 1.0608 | 0.2807 | 5405 | 13.6 | 17.0 | ff210cd10218 | verified |
| Random-key dropout | 2028 | `track2_croco_abl_randkey_s2028` | 1.0868 | 0.2906 | 5405 | 6.3 | 17.0 | 8526aa7f0c5a | verified |
| - L_delta | 2026 | `track2_croco_abl_no_delta` | 0.9493 | 0.2788 | 5405 | 7.7 | 17.0 | a4fdc628f980 | verified |
| - L_delta | 2027 | `track2_croco_abl_no_delta_seed2` | 0.9811 | 0.2820 | 5405 | 4.6 | 17.0 | 9c0353a37f3c | verified |
| - L_delta | 2028 | `track2_croco_abl_no_delta_s2028` | 0.8424 | 0.2835 | 5405 | 6.1 | 17.0 | bbf87dc8934d | verified |
| Image-plane mask transport | 2026 | `track2_croco_abl_image_plane` | 0.8578 | 0.2911 | 5405 | 6.5 | 17.0 | 9954a269a257 | verified |
| Image-plane mask transport | 2027 | `track2_croco_abl_image_plane_seed2` | 0.8801 | 0.2875 | 5405 | 12.7 | 17.0 | 83a1538a9f04 | verified |
| Image-plane mask transport | 2028 | `track2_croco_abl_image_plane_s2028` | 0.7421 | 0.2943 | 5405 | 6.1 | 17.0 | afacbaf27f03 | verified |
| - teacher / anchor | 2026 | `track2_croco_abl_no_teacher` | 0.9206 | 0.3170 | 5405 | 6.5 | 15.4 | 3fa7cb9093c5 | verified |
| - teacher / anchor | 2027 | `track2_croco_abl_no_teacher_seed2` | 0.7378 | 0.3014 | 5405 | 9.2 | 15.4 | ef24e477ff39 | verified |
| - teacher / anchor | 2028 | `track2_croco_abl_no_teacher_s2028` | 1.3630 | 0.3010 | 5405 | 4.0 | 15.4 | e99dab89bfff | verified |
| - monocular branch (= '+ corr. dropout + overlays') | 2026 | `track2_croco_x6b_x4_corrdrop` | 0.7664 | 0.2870 | 5405 | 6.2 | 16.3 | 3af27fa68b4a | verified |
| - monocular branch (= '+ corr. dropout + overlays') | 2027 | `track2_croco_abl_no_mono_seed2` | 0.6883 | 0.2819 | 5405 | 4.9 | 16.3 | 898cc5039baf | verified |
| - monocular branch (= '+ corr. dropout + overlays') | 2028 | `track2_croco_abl_no_mono_s2028` | 0.6267 | 0.2890 | 5405 | 4.7 | 16.3 | e6dc70461185 | verified |
| - sensor overlays | 2026 | `track2_croco_abl_no_overlay` | 0.7736 | 0.2932 | 5405 | 6.2 | 17.0 | 6b281bb06851 | verified |
| - sensor overlays | 2027 | `track2_croco_abl_no_overlay_seed2` | 0.6580 | 0.2853 | 5405 | 6.8 | 17.0 | 30b3b3c27473 | verified |
| - sensor overlays | 2028 | `track2_croco_abl_no_overlay_s2028` | 0.8669 | 0.2861 | 5405 | 4.6 | 17.0 | ffefcd340809 | verified |
| CorrMask p_mask = 0 | 2026 | `track2_croco_cm_p000_s2026` | 0.8525 | 0.2887 | 5405 | 5.1 | 17.0 | 0231da197c39 | verified |
| CorrMask p_mask = 0 | 2027 | `track2_croco_cm_p000_s2027` | 0.6150 | 0.2847 | 5405 | 5.1 | 17.0 | c58984737aaa | verified |
| CorrMask p_mask = 0 | 2028 | `track2_croco_cm_p000_s2028` | 0.8068 | 0.2906 | 5405 | 5.6 | 17.0 | 6e012a93ad51 | verified |
| CorrMask p_mask = 0.10 | 2026 | `track2_croco_cm_p010_s2026` | 0.9461 | 0.2870 | 5405 | 4.5 | 17.0 | 7bd006379424 | verified |
| CorrMask p_mask = 0.10 | 2027 | `track2_croco_cm_p010_s2027` | 0.7775 | 0.2822 | 5405 | 4.5 | 17.0 | 8f23c6fafbb9 | verified |
| CorrMask p_mask = 0.10 | 2028 | `track2_croco_cm_p010_s2028` | 0.7027 | 0.2906 | 5405 | 4.6 | 17.0 | 60f6cd171c0f | verified |
| CorrMask p_mask = 0.50 | 2026 | `track2_croco_cm_p050_s2026` | 0.6332 | 0.2836 | 5405 | 4.5 | 17.0 | 4e67abd33f21 | verified |
| CorrMask p_mask = 0.50 | 2027 | `track2_croco_cm_p050_s2027` | 0.7245 | 0.2814 | 5405 | 4.6 | 17.0 | 2b45d3524f71 | verified |
| CorrMask p_mask = 0.50 | 2028 | `track2_croco_cm_p050_s2028` | 0.7350 | 0.2932 | 5405 | 4.5 | 17.0 | 2362fcb91c9b | verified |
| lambda_delta = 0.5 | 2026 | `track2_croco_ld050_s2026` | 0.7898 | 0.2825 | 5405 | 4.5 | 17.0 | 442207463dd1 | verified |
| lambda_delta = 0.5 | 2027 | `track2_croco_ld050_s2027` | 0.8075 | 0.2822 | 5405 | 4.5 | 17.0 | 5f862a261772 | verified |
| lambda_delta = 0.5 | 2028 | `track2_croco_ld050_s2028` | 0.7035 | 0.2889 | 5405 | 4.5 | 17.0 | 9268de54da85 | verified |
| lambda_delta = 2.0 | 2026 | `track2_croco_ld200_s2026` | 0.8412 | 0.2919 | 5405 | 4.5 | 17.0 | 7e8f48b02d5b | verified |
| lambda_delta = 2.0 | 2027 | `track2_croco_ld200_s2027` | 0.7176 | 0.2937 | 5405 | 6.1 | 17.0 | 1f4bee539a68 | verified |
| lambda_delta = 2.0 | 2028 | `track2_croco_ld200_s2028` | 1.3689 | 0.7010 | 5405 | 4.4 | 17.0 | a15cb03dff0a | verified |
| lambda_corr = 0.125 | 2026 | `track2_croco_lc0125_s2026` | 0.7903 | 0.2868 | 5405 | 4.7 | 17.0 | 936485d84f77 | verified |
| lambda_corr = 0.125 | 2027 | `track2_croco_lc0125_s2027` | 0.7915 | 0.2861 | 5405 | 4.6 | 17.0 | ba82105ae2b4 | verified |
| lambda_corr = 0.125 | 2028 | `track2_croco_lc0125_s2028` | 0.7509 | 0.2909 | 5405 | 4.4 | 17.0 | d76222189dd8 | verified |
| lambda_corr = 0.5 | 2026 | `track2_croco_lc050_s2026` | 0.7860 | 0.2855 | 5405 | 5.0 | 17.0 | d4e1092498c7 | verified |
| lambda_corr = 0.5 | 2027 | `track2_croco_lc050_s2027` | 0.7673 | 0.2869 | 5405 | 4.8 | 17.0 | 6d69eb1f0dd3 | verified |
| lambda_corr = 0.5 | 2028 | `track2_croco_lc050_s2028` | 0.7078 | 0.2898 | 5405 | 5.7 | 17.0 | 1331477e18b0 | verified |
| Clean-only baseline | 2026 | `track2_e0_croco_clean` | 2.1951 | 0.2815 | 5405 | 4.3 | 22.6 | c5e15131a150 | verified |
| Clean-only baseline | 2027 | `track2_e0_croco_clean_s2027` | 1.5312 | 0.2854 | 5405 | 1.9 | 22.6 | 7ac0fa3767dc | verified |
| Clean-only baseline | 2028 | `track2_e0_croco_clean_s2028` | 1.2475 | 0.2858 | 5405 | 2.0 | 22.6 | b42e4df549ef | verified |
| Clean-corrupted consistency | 2026 | `track2_croco_x4_robust_from_start` | 1.4872 | 0.2801 | 5405 | 10.0 | 24.3 | fc017759ab49 | verified |
| Clean-corrupted consistency | 2027 | `track2_croco_x4_robust_from_start_s2027` | 1.5079 | 0.2802 | 5405 | 4.3 | 24.3 | 5a83dbaa86c5 | verified |
| Clean-corrupted consistency | 2028 | `track2_croco_x4_robust_from_start_s2028` | 0.7364 | 0.2881 | 5405 | 4.6 | 24.3 | 37bf731213d2 | verified |
| Short robust fine-tuning | 2026 | `track2_croco_x2_e4_fix` | 1.6782 | 0.2807 | 1351 | 1.5 | 13.3 | 46b702016ceb | verified |
| Short robust fine-tuning | 2027 | `track2_croco_x2_e4_fix_s2027` | 1.5150 | 0.2826 | 1351 | 0.5 | 13.3 | 473547e66420 | verified |
| Short robust fine-tuning | 2028 | `track2_croco_x2_e4_fix_s2028` | 1.8684 | 0.2817 | 1351 | 0.6 | 13.3 | 63cef72c0d67 | verified |
| Reliability gating | 2026 | `track2_croco_x6a_x4_fallback` | 1.0743 | 0.2878 | 5405 | 9.5 | 16.3 | 2a04de3b98af | verified |
| Reliability gating | 2027 | `track2_croco_x6a_x4_fallback_s2027` | 0.7858 | 0.2808 | 5405 | 4.4 | 16.3 | 44300d121a1e | verified |
| Reliability gating | 2028 | `track2_croco_x6a_x4_fallback_s2028` | 0.7997 | 0.2898 | 5405 | 4.7 | 16.3 | 6cc0c02a2865 | verified |
| Attention supervision | 2026 | `track2_croco_x5b_x4_attnsup_weak` | 1.2747 | 0.2854 | 5405 | 9.3 | 16.4 | 60cdb3769a75 | verified |
| Attention supervision | 2027 | `track2_croco_x5b_x4_attnsup_weak_s2027` | 1.3756 | 0.2944 | 5405 | 4.3 | 16.3 | 3b855e57b719 | verified |
| Attention supervision | 2028 | `track2_croco_x5b_x4_attnsup_weak_s2028` | 1.4256 | 0.2901 | 5405 | 5.4 | 16.4 | 0ece7f881e8f | verified |
| Attention supervision + corr. dropout | 2026 | `track2_croco_x9_x6b_attnsup` | 0.9274 | 0.2899 | 5405 | 8.6 | 16.4 | 6bc05f2f266d | verified |
| Attention supervision + corr. dropout | 2027 | `track2_croco_x9_x6b_attnsup_s2027` | 0.9976 | 0.2831 | 5405 | 4.3 | 16.4 | b5f144f4cdaa | verified |
| Attention supervision + corr. dropout | 2028 | `track2_croco_x9_x6b_attnsup_s2028` | 0.9730 | 0.2930 | 5405 | 4.4 | 16.4 | 381a7928e08d | verified |

63 of 63 runs verified. Total training time of the listed runs: 357 GPU-hours (median 4.7 h per run); evaluation of one checkpoint on the robust20 suite takes about one further GPU-hour.

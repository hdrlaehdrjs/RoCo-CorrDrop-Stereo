# Configuration check of the three-seed experiments

| variant | seed | config | difference to the seed-2026 config | difference to the full model | ok |
|---|---|---|---|---|---|
| Full model | 2026 | `track2_croco_x7_x6b_mono` | none | – | yes |
| Full model | 2027 | `track2_croco_abl_seed2` | none | – | yes |
| Full model | 2028 | `track2_croco_abl_seed3` | none | – | yes |
| - correspondence dropout | 2026 | `track2_croco_abl_no_dropout` | none | `corrdrop.p`: 0.5 → 0.0 | yes |
| - correspondence dropout | 2027 | `track2_croco_abl_no_dropout_seed2` | none | `corrdrop.p`: 0.5 → 0.0 | yes |
| - correspondence dropout | 2028 | `track2_croco_abl_no_dropout_s2028` | none | `corrdrop.p`: 0.5 → 0.0 | yes |
| Random-key dropout | 2026 | `track2_croco_abl_randkey_s1` | none | `corrdrop.keys`: None → random | yes |
| Random-key dropout | 2027 | `track2_croco_abl_randkey_s2` | none | `corrdrop.keys`: None → random | yes |
| Random-key dropout | 2028 | `track2_croco_abl_randkey_s2028` | none | `corrdrop.keys`: None → random | yes |
| - L_delta | 2026 | `track2_croco_abl_no_delta` | none | `loss.lambda_delta`: 1.0 → 0.0 | yes |
| - L_delta | 2027 | `track2_croco_abl_no_delta_seed2` | none | `loss.lambda_delta`: 1.0 → 0.0 | yes |
| - L_delta | 2028 | `track2_croco_abl_no_delta_s2028` | none | `loss.lambda_delta`: 1.0 → 0.0 | yes |
| Image-plane mask transport | 2026 | `track2_croco_abl_image_plane` | none | `s2aug.transport`: zbuffer → image_plane | yes |
| Image-plane mask transport | 2027 | `track2_croco_abl_image_plane_seed2` | none | `s2aug.transport`: zbuffer → image_plane | yes |
| Image-plane mask transport | 2028 | `track2_croco_abl_image_plane_s2028` | none | `s2aug.transport`: zbuffer → image_plane | yes |
| - teacher / anchor | 2026 | `track2_croco_abl_no_teacher` | none | `corrmask.provider`: teacher_confidence → gt_priority; `loss.delta_weight`: teacher → valid; `loss.lambda_anchor`: 0.1 → 0.0 | yes |
| - teacher / anchor | 2027 | `track2_croco_abl_no_teacher_seed2` | none | `corrmask.provider`: teacher_confidence → gt_priority; `loss.delta_weight`: teacher → valid; `loss.lambda_anchor`: 0.1 → 0.0 | yes |
| - teacher / anchor | 2028 | `track2_croco_abl_no_teacher_s2028` | none | `corrmask.provider`: teacher_confidence → gt_priority; `loss.delta_weight`: teacher → valid; `loss.lambda_anchor`: 0.1 → 0.0 | yes |
| - monocular branch (= '+ corr. dropout + overlays') | 2026 | `track2_croco_x6b_x4_corrdrop` | none | `mono.fuse_bias_init`: 6.0 → None; `mono.fuse_hidden`: 16 → None; `mono.hooks`: [5, 11, 17, 23] → None; `mono.init_from_stereo_head`: True → None; `mono.lambda_mono`: 0.5 → None; `mono.lr_scale`: 10 → None | yes |
| - monocular branch (= '+ corr. dropout + overlays') | 2027 | `track2_croco_abl_no_mono_seed2` | none | `mono.fuse_bias_init`: 6.0 → None; `mono.fuse_hidden`: 16 → None; `mono.hooks`: [5, 11, 17, 23] → None; `mono.init_from_stereo_head`: True → None; `mono.lambda_mono`: 0.5 → None; `mono.lr_scale`: 10 → None | yes |
| - monocular branch (= '+ corr. dropout + overlays') | 2028 | `track2_croco_abl_no_mono_s2028` | none | `mono.fuse_bias_init`: 6.0 → None; `mono.fuse_hidden`: 16 → None; `mono.hooks`: [5, 11, 17, 23] → None; `mono.init_from_stereo_head`: True → None; `mono.lambda_mono`: 0.5 → None; `mono.lr_scale`: 10 → None | yes |
| - sensor overlays | 2026 | `track2_croco_abl_no_overlay` | none | `s2aug.p_overlay`: 0.3 → 0.0 | yes |
| - sensor overlays | 2027 | `track2_croco_abl_no_overlay_seed2` | none | `s2aug.p_overlay`: 0.3 → 0.0 | yes |
| - sensor overlays | 2028 | `track2_croco_abl_no_overlay_s2028` | none | `s2aug.p_overlay`: 0.3 → 0.0 | yes |
| CorrMask p_mask = 0 | 2026 | `track2_croco_cm_p000_s2026` | none | `s2aug.p_mask`: 0.25 → 0.0 | yes |
| CorrMask p_mask = 0 | 2027 | `track2_croco_cm_p000_s2027` | none | `s2aug.p_mask`: 0.25 → 0.0 | yes |
| CorrMask p_mask = 0 | 2028 | `track2_croco_cm_p000_s2028` | none | `s2aug.p_mask`: 0.25 → 0.0 | yes |
| CorrMask p_mask = 0.10 | 2026 | `track2_croco_cm_p010_s2026` | none | `s2aug.p_mask`: 0.25 → 0.1 | yes |
| CorrMask p_mask = 0.10 | 2027 | `track2_croco_cm_p010_s2027` | none | `s2aug.p_mask`: 0.25 → 0.1 | yes |
| CorrMask p_mask = 0.10 | 2028 | `track2_croco_cm_p010_s2028` | none | `s2aug.p_mask`: 0.25 → 0.1 | yes |
| CorrMask p_mask = 0.50 | 2026 | `track2_croco_cm_p050_s2026` | none | `s2aug.p_mask`: 0.25 → 0.5 | yes |
| CorrMask p_mask = 0.50 | 2027 | `track2_croco_cm_p050_s2027` | none | `s2aug.p_mask`: 0.25 → 0.5 | yes |
| CorrMask p_mask = 0.50 | 2028 | `track2_croco_cm_p050_s2028` | none | `s2aug.p_mask`: 0.25 → 0.5 | yes |
| lambda_delta = 0.5 | 2026 | `track2_croco_ld050_s2026` | none | `loss.lambda_delta`: 1.0 → 0.5 | yes |
| lambda_delta = 0.5 | 2027 | `track2_croco_ld050_s2027` | none | `loss.lambda_delta`: 1.0 → 0.5 | yes |
| lambda_delta = 0.5 | 2028 | `track2_croco_ld050_s2028` | none | `loss.lambda_delta`: 1.0 → 0.5 | yes |
| lambda_delta = 2.0 | 2026 | `track2_croco_ld200_s2026` | none | `loss.lambda_delta`: 1.0 → 2.0 | yes |
| lambda_delta = 2.0 | 2027 | `track2_croco_ld200_s2027` | none | `loss.lambda_delta`: 1.0 → 2.0 | yes |
| lambda_delta = 2.0 | 2028 | `track2_croco_ld200_s2028` | none | `loss.lambda_delta`: 1.0 → 2.0 | yes |
| lambda_corr = 0.125 | 2026 | `track2_croco_lc0125_s2026` | none | `loss.lambda_corr`: 0.25 → 0.125 | yes |
| lambda_corr = 0.125 | 2027 | `track2_croco_lc0125_s2027` | none | `loss.lambda_corr`: 0.25 → 0.125 | yes |
| lambda_corr = 0.125 | 2028 | `track2_croco_lc0125_s2028` | none | `loss.lambda_corr`: 0.25 → 0.125 | yes |
| lambda_corr = 0.5 | 2026 | `track2_croco_lc050_s2026` | none | `loss.lambda_corr`: 0.25 → 0.5 | yes |
| lambda_corr = 0.5 | 2027 | `track2_croco_lc050_s2027` | none | `loss.lambda_corr`: 0.25 → 0.5 | yes |
| lambda_corr = 0.5 | 2028 | `track2_croco_lc050_s2028` | none | `loss.lambda_corr`: 0.25 → 0.5 | yes |
| Clean-only baseline | 2026 | `track2_e0_croco_clean` | none | (different recipe) | yes |
| Clean-only baseline | 2027 | `track2_e0_croco_clean_s2027` | none | (different recipe) | yes |
| Clean-only baseline | 2028 | `track2_e0_croco_clean_s2028` | none | (different recipe) | yes |
| Clean-corrupted consistency | 2026 | `track2_croco_x4_robust_from_start` | none | (different recipe) | yes |
| Clean-corrupted consistency | 2027 | `track2_croco_x4_robust_from_start_s2027` | none | (different recipe) | yes |
| Clean-corrupted consistency | 2028 | `track2_croco_x4_robust_from_start_s2028` | none | (different recipe) | yes |
| Short robust fine-tuning | 2026 | `track2_croco_x2_e4_fix` | none | (different recipe) | yes |
| Short robust fine-tuning | 2027 | `track2_croco_x2_e4_fix_s2027` | none | (different recipe) | yes |
| Short robust fine-tuning | 2028 | `track2_croco_x2_e4_fix_s2028` | none | (different recipe) | yes |
| Reliability gating | 2026 | `track2_croco_x6a_x4_fallback` | none | (different recipe) | yes |
| Reliability gating | 2027 | `track2_croco_x6a_x4_fallback_s2027` | none | (different recipe) | yes |
| Reliability gating | 2028 | `track2_croco_x6a_x4_fallback_s2028` | none | (different recipe) | yes |
| Attention supervision | 2026 | `track2_croco_x5b_x4_attnsup_weak` | none | (different recipe) | yes |
| Attention supervision | 2027 | `track2_croco_x5b_x4_attnsup_weak_s2027` | none | (different recipe) | yes |
| Attention supervision | 2028 | `track2_croco_x5b_x4_attnsup_weak_s2028` | none | (different recipe) | yes |
| Attention supervision + corr. dropout | 2026 | `track2_croco_x9_x6b_attnsup` | none | (different recipe) | yes |
| Attention supervision + corr. dropout | 2027 | `track2_croco_x9_x6b_attnsup_s2027` | none | (different recipe) | yes |
| Attention supervision + corr. dropout | 2028 | `track2_croco_x9_x6b_attnsup_s2028` | none | (different recipe) | yes |

63 configurations checked, 0 problems.

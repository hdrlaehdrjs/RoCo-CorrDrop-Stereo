# RoCo-19 – KoreaU_DLmath · RoCo-Spring Track 2 (stereo) · reproducibility package

Code for the leaderboard entry **RoCo-19-CorrDrop-Stereo** and for every table / figure of the workshop paper.
The repository contains the final method, the paper ablations and exploratory variants (three seeds each) and selected
development utilities; non-paper variants (e.g. the frozen-backbone CARE variants in `track2/care.py`) are disabled by
default and not used by the reported model.

## 1. Method in one paragraph
CroCo-Stereo v2 (ViT-L encoder, ViT-B decoder, DPT head; 437.4 M parameters) is fine-tuned on Spring with a
clean/corrupted **twin** forward. The corrupted branch sees S2Aug (7 generic corruption families, stereo-consistent,
occluder masks transported to the right view with a disparity z-buffer), CorrMask (teacher-confidence-guided box
occlusions), sensor overlays, and **correspondence dropout**: for a random rectangular region of left queries, the right-view
keys within ±1 token row (the epipolar band) are masked in all 12 decoder cross-attention layers, so the prediction must
come from context. A monocular DPT head on the detached left encoder tokens (blocks 5/11/17/23) and a per-pixel fuser
(+20.4 M parameters) are trained jointly; the fused prediction is `d = w d_stereo + (1 - w) d_mono`.

Objective per step (`scripts/train_track2.py`, weights from `configs/track2/track2_croco_x2_e4_fix.yaml` and
`track2_croco_x7_x6b_mono.yaml`):

    L = L_clean + 0.1 L_anchor + 0.5 L_mono(clean)
        + eta(t) [0.25 L_corr + 1.0 L_delta] + 0.5 L_mono(corrupted),      eta(t) = min(1, t / (0.1 T))

`L_clean` / `L_corr` are the CroCo Laplacian loss of the fused prediction on the clean / corrupted input, `L_delta` the
twin consistency between the two predictions, `L_anchor` the anchor to the frozen clean-finetuned teacher and `L_mono`
the Laplacian loss of the monocular head alone. `eta` ramps the corruption terms linearly over the first 10 % of the
steps (S2Aug probability / severity follow their own 10 % curriculum). Fog, frost, speckle noise, zoom blur and elastic
transform are never used in training.

**Inference** uses a single fixed checkpoint containing the stereo network, the monocular head and the fusion module
(tiles 352×704, overlap 0.7, fp32). The teacher, the image-space corruptions, CorrMask and correspondence dropout are
training-only. There is no test-time adaptation, corruption detection, ensembling or corruption-specific branch
switching. The learned fusion is strongly stereo-dominant: over the three full-model checkpoints and all 21 local
conditions the stereo weight never falls below 0.98, and disabling the monocular module at inference changes the mean
ΔAbs by less than 10⁻⁴ px (`results/three_seed/mono/`).

## 2. Layout
| path | content |
|---|---|
| `track2/` | library: `croco_model.py` (wrapper, tiled inference), `care.py` (correspondence dropout, random-key control), `mono.py` (mono head + fuser), `s2aug.py`, `stereo_corrmask.py`, `twin_data.py`, `losses.py`, `backbones.py`, `robust20.py` (local corruption suite), `val_suite.py` (metrics), `config.py`, `paths.py` |
| `scripts/train_track2.py` | training entry point (writes `contract.json` with config, seed, source and checkpoint hashes) |
| `scripts/eval_suite.py` | evaluation on the frozen local suite (`--suite robust20`) |
| `scripts/build_robust20_val.py` | builds the frozen 20-corruption validation cache (severity 3) |
| `scripts/run_train_eval.sh` | `<gpu> <config> <name>`: train + robust20 evaluation of the final checkpoint |
| `scripts/make_submission.py`, `run_submission.sh` | test predictions (21 conditions × 1000 frames × 2 views) → official subsampling binaries → the two upload files |
| `scripts/three_seed/` | the three-seed experiments behind the local tables of the paper: `runs.json` (variant × seed → config and run folder), `check_configs.py`, `summarize.py` (run verification + tables), `regional.py`, `mono_same_checkpoint.py`, `latency.py`, `latex_tables.py` |
| `scripts/final_seed_summary.py`, `final_targeted_summary.py` | seed / ablation tables at the time of the paper submission (two seeds for most ablations) |
| `scripts/final_analysis.py`, `analyze_evidence_free.py`, `figures/` | regional (sky / textureless) analysis, fusion weights, latency, qualitative figures |
| `configs/track2/` | the final configs and their inheritance chain (32 files), `seeds/` (19 seed repetitions), `sensitivity/` (21 CorrMask-probability and loss-weight settings) |
| `splits/` | scene-disjoint split: 4054 train pairs, 49 validation frames (28 primary: scenes 0038/0039/0045/0047), `track2_train_all.txt` (4103 pairs, submission model), corruption seeds |
| `tests/` | CPU test suite (`python3 -m pytest -q tests`); `gpu_croco_tiling_parity.py` = optional GPU parity check of the tiled inference |
| `results/` | every local table of the paper as csv/md (`results/README.md` maps tables to files; `results/three_seed/` = three-seed tables, regional and monocular analyses), appendix figure, submission bundle hashes |
| `third_party/croco-source.json` | pinned CroCo commit |

Configs: `track2_croco_x7_x6b_mono` = full model (local protocol, seed 2026); `track2_croco_abl_seed2/3` = seeds 2027/2028;
`track2_croco_final_submit` = leaderboard model (all 4103 pairs, init = official Spring-finetuned CroCo-Stereo);
`track2_croco_abl_*` = ablations (each differs from the full config only in the ablated key — enforced by a unit test);
`track2_croco_abl_randkey_s1/s2` = budget-matched random-key dropout control; `track2_e0_croco_clean` = clean-only baseline E0 / teacher;
`track2_croco_x4_robust_from_start` = clean–corrupted consistency stage; `track2_croco_x6b_x4_corrdrop` = `+ corr. dropout + overlays`
(identical to the no-monocular ablation); `track2_croco_x2_e4_fix` = short robust fine-tuning; `track2_croco_x6a_x4_fallback`,
`track2_croco_x5b_x4_attnsup_weak`, `track2_croco_x9_x6b_attnsup` = exploratory variants (reliability gating, attention supervision).
`seeds/*_s2027`, `*_s2028` repeat a recipe with another seed; `sensitivity/` holds the CorrMask-probability (`cm_p*`) and loss-weight
(`ld*`, `lc*`) settings, each differing from the full model in one coefficient. Configs are found by name in the sub-folders.
`scripts/three_seed/runs.json` lists the config of every variant and seed.

## 3. Setup
Tested with Python 3.11, PyTorch 2.7.1 + CUDA 11.8 on NVIDIA L40S (48 GB). One 48 GB GPU is enough (training peak
17 GB with `micro_batch: 1`; ≈ 5 h per run).

```bash
git clone https://github.com/hdrlaehdrjs/RoCo-CorrDrop-Stereo && cd RoCo-CorrDrop-Stereo
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install --no-deps imagecorruptions==1.1.2        # only its frost texture images are used (track2/robust20.py)

# CroCo (not redistributed; pinned commit, see third_party/croco-source.json)
git clone https://github.com/naver/croco third_party/croco
git -C third_party/croco checkout 5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb
(cd third_party/croco/models/curope && python setup.py build_ext --inplace)   # cuRoPE kernels (CroCo falls back to a slower RoPE without them)

# official CroCo-Stereo weights (download script of the CroCo repository)
mkdir -p checkpoints/croco_official
for m in crocostereo.pth crocostereo_finetune_spring.pth; do
  wget https://download.europe.naverlabs.com/ComputerVision/CroCo/StereoFlow_models/$m -P checkpoints/croco_official/
done
sha256sum checkpoints/croco_official/*.pth
#   8b48935772c98714db1a18773d687da273d3ab721410cc5e398451a0dac8982f  crocostereo.pth
#   123273a722a58134c6816efe5122c427815c92f82896be4d484e6d21b52da575  crocostereo_finetune_spring.pth
```

Data and external tools are located through `track2/paths.py`; defaults are relative to the repository root and every
one can be overridden by an environment variable:

| variable | default | needed for |
|---|---|---|
| `ROCO_TRACK2_WORK` | repository root | holds `checkpoints/`, `outputs/`, `submissions/` |
| `ROCO_SPRING` | `data/spring` | Spring dataset (`train/`, `test/`), all training / evaluation |
| `ROCO_CROCO_ROOT` | `$ROCO_TRACK2_WORK/third_party/croco` | CroCo checkout |
| `ROCO_DATA` | `data/` | RobustSpring test corruptions (submission only) |
| `ROCO_SUBSAMPLING_TOOLS` | `third_party/subsampling_tools` | official Spring binaries `disp1_subsampling`, `disp1_robust_subsampling` (submission only) |
| `ROCO_TRACK3_ROOT` | `third_party/track3` | optional: DEFOM backbone and Q4Aug source-parity tests (skipped when absent) |

Spring / RobustSpring and the subsampling binaries must be obtained from their providers (see `THIRD_PARTY_NOTICES.md`).

## 4. Reproduce
```bash
python3 -m pytest -q tests                                                       # CPU tests (Track 3-dependent ones skip)
python3 scripts/build_robust20_val.py                                            # frozen local robust20 cache (49 frames × 20)
# train + robust20 evaluation of one config -> outputs/track2/final_runs/<name>/{train,eval}; use the config name as run name
# clean-only baseline, seed 2026 = frozen teacher of all robust runs
scripts/run_train_eval.sh 0 track2_e0_croco_clean track2_e0_croco_clean
mkdir -p checkpoints/track2 && cp outputs/track2/final_runs/track2_e0_croco_clean/train/final.pt checkpoints/track2/croco_clean_teacher.pt
scripts/run_train_eval.sh 0 track2_croco_x7_x6b_mono track2_croco_x7_x6b_mono        # full model, seed 2026
scripts/run_train_eval.sh 1 track2_croco_abl_seed2 track2_croco_abl_seed2            # full model, seed 2027
scripts/run_train_eval.sh 2 track2_croco_abl_no_dropout track2_croco_abl_no_dropout  # … any other config
# all 63 runs of the three-seed tables (the first name printed is the clean-only baseline trained above)
python3 scripts/three_seed/check_configs.py                                      # consistency of the 63 configs, no run needed
for c in $(python3 scripts/three_seed/common.py | tail -n +2); do scripts/run_train_eval.sh 0 $c $c; done
python3 scripts/three_seed/summarize.py                                          # verification + tables -> outputs/track2/three_seed/
CUDA_VISIBLE_DEVICES=0 python3 scripts/three_seed/regional.py predict e0:2026 full:2026 e0:2027 full:2027 e0:2028 full:2028
python3 scripts/three_seed/regional.py summarize                                 # sky / textured-region statistics
for s in 2026 2027 2028; do CUDA_VISIBLE_DEVICES=0 python3 scripts/three_seed/mono_same_checkpoint.py run --seed $s; done
python3 scripts/three_seed/mono_same_checkpoint.py summarize                     # fused vs stereo-only inference of the same checkpoint
CUDA_VISIBLE_DEVICES=0 python3 scripts/three_seed/latency.py --gpu-index 0       # inference overhead (use an idle GPU)
python3 scripts/three_seed/latex_tables.py                                       # LaTeX bodies of the local tables
# leaderboard model and submission files
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_track2.py --config track2_croco_final_submit --out outputs/track2/final_submit
scripts/run_submission.sh outputs/track2/final_submit/final.pt final_x7 "0 1 2 3"
#   -> submissions/track2/final_x7/bundle/clean/disp1_submission.hdf5   (~141 MB)
#      submissions/track2/final_x7/bundle/robust/disp1_robustness.hdf5  (~161 MB, 21 groups)
```
Upload only those two files (raw predictions are ~250 GB and can be deleted after bundling). `train_track2.py` refuses
to write into an existing `--out` folder.

Every robust variant, for every seed, uses the seed-2026 clean-only checkpoint as its frozen teacher
(`checkpoints/track2/croco_clean_teacher.pt`); the short fine-tuning variant also starts from it. The clean-only
baseline and the consistency stage use `micro_batch: 2` (peak 23–24 GB instead of 17 GB).

**Tables and figures.** The committed files in `results/` are the outputs of the runs above (`results/README.md` maps
each table of the paper to its file). The scripts in `scripts/three_seed/` look for a run in
`outputs/track2/final_runs/<config>/` first and otherwise in the folder recorded in `runs.json` (our original run
names under `outputs/track2/`). The older summary scripts (`scripts/final_seed_summary.py`,
`final_targeted_summary.py`) read the evaluation folders listed in their `RUNS` tables; adapt `RUNS` to your folder
names. The seed-2026 regional / fusion-weight analyses and the qualitative figures need model predictions that are not
committed:

    checkpoints (E0 + full model, paths in MODELS of scripts/final_analysis.py)
      -> python3 scripts/final_analysis.py predict      (prediction cache outputs/track2/final_analysis/cache, GPU)
      -> python3 scripts/final_analysis.py stats | figures | latency
      -> python3 scripts/figures/qualitative_robustness_appendix.py figure

The final appendix figure itself is committed in `results/figures/`.

## 5. Results
Official [Spring stereo benchmark](https://spring-benchmark.org/stereo) entry **RoCo-19-CorrDrop-Stereo**
(model `track2_croco_final_submit`):

| Abs | 1px | D1 |
|---|---|---|
| 0.457 | 7.095 | 2.615 |

Robustness columns of the same leaderboard:

| ΔAbs | Δ1px | ΔD1 |
|---|---|---|
| 1.441 | 18.088 | 1.909 |

See the leaderboard for the per-region breakdown and metric definitions.

Local protocol (28 primary validation frames, 20 corruptions, mean ΔAbs; three independently trained runs per variant
with seeds 2026 / 2027 / 2028; mean ± sample sd):

| variant | ΔAbs (individual runs) | ΔAbs | clean Abs |
|---|---|---|---|
| clean-only baseline | 2.195 / 1.531 / 1.247 | 1.658 ± 0.486 | 0.284 |
| clean–corrupted consistency | 1.487 / 1.508 / 0.736 | 1.244 ± 0.440 | 0.283 |
| + corr. dropout + overlays (= no monocular branch) | 0.766 / 0.688 / 0.627 | 0.694 ± 0.070 | 0.286 |
| **full** | 0.656 / 0.783 / 0.570 | **0.670 ± 0.107** | 0.287 |
| no correspondence dropout | 0.894 / 0.839 / 0.946 | 0.893 ± 0.054 | 0.289 |
| random-key dropout (budget-matched control) | 1.385 / 1.061 / 1.087 | 1.177 ± 0.180 | 0.287 |
| no L_delta | 0.949 / 0.981 / 0.842 | 0.924 ± 0.073 | 0.281 |
| image-plane mask transport | 0.858 / 0.880 / 0.742 | 0.827 ± 0.074 | 0.291 |
| no teacher / anchor | 0.921 / 0.738 / 1.363 | 1.007 ± 0.321 | 0.306 |
| no sensor overlay | 0.774 / 0.658 / 0.867 | 0.766 ± 0.105 | 0.288 |
| CorrMask p_mask = 0 / 0.10 / 0.50 (default 0.25) | – | 0.758 ± 0.126 / 0.809 ± 0.125 / 0.698 ± 0.056 | 0.288 / 0.287 / 0.286 |
| λ_Δ = 0.5 / 2.0 (default 1.0) | – | 0.767 ± 0.056 / 0.976 ± 0.346 | 0.285 / 0.429 |
| λ_corr = 0.125 / 0.5 (default 0.25) | – | 0.778 ± 0.023 / 0.754 ± 0.041 | 0.288 / 0.287 |
| short robust fine-tuning | 1.678 / 1.515 / 1.868 | 1.687 ± 0.177 | 0.282 |
| reliability gating | 1.074 / 0.786 / 0.800 | 0.887 ± 0.163 | 0.286 |
| attention supervision | 1.275 / 1.376 / 1.426 | 1.359 ± 0.077 | 0.290 |
| attention supervision + corr. dropout | 0.927 / 0.998 / 0.973 | 0.966 ± 0.036 | 0.289 |

These are descriptive statistics of three runs; no significance is claimed. The clean-only baseline and the
consistency stage vary strongly between seeds, the full model much less; every full-model run is below every
clean-only run. One of the three λ_Δ = 2.0 runs became unstable after warm-up (clean Abs 0.701), which explains the
mean clean Abs of that row. Individual runs, per-corruption values, matched-seed differences, regional statistics and
the same-checkpoint monocular ablation: `results/three_seed/` (index in `results/README.md`).

## 6. Declared data, checkpoints and compute
Training data: Spring train split only (scene-disjoint local split for the paper, all 4103 pairs for the leaderboard
model). Initialisation: official CroCo-Stereo weights (public pre-Spring weights for local runs, the official
Spring-finetuned weights for the leaderboard model, which also serve as its frozen teacher). No RobustSpring test image,
label or leaderboard feedback was used for training or model selection; the same checkpoint and settings produced both
uploads (`results/BUNDLE_CONTRACT.json`: checkpoint sha256 3e579178…1d42). The local corruption suite is a
re-implementation (ImageNet-C style + depth-aware fog), not the RobustSpring generator. No git on the training host: every
run records the sha256 of its sources in `contract.json`. Hardware: NVIDIA L40S (48 GB), one GPU per run; a full
training run takes ≈ 5 GPU-hours and peaks at 17 GB, the robust20 evaluation of a checkpoint about one further
GPU-hour. The 63 local runs behind the three-seed tables took 357 GPU-hours of training in total; training time,
peak memory and checkpoint hash of every run are listed in `results/three_seed/runs.md`.

## 7. Licences
Code written for this repository: CC BY-NC-SA 4.0 (`LICENSE`), matching CroCo / CroCo-Stereo, which it builds on and
whose tiled inference and loss it re-implements. Non-commercial use only. External code, checkpoints and datasets
are not redistributed; their terms are listed in `THIRD_PARTY_NOTICES.md`.

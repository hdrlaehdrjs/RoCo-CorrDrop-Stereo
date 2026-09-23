# RoCo-19 – KoreaU_DLmath · RoCo-Spring Track 2 (stereo) · reproducibility package

Code for the leaderboard entry **RoCo-19-CorrDrop-Stereo** and for every table / figure of the workshop paper.
The repository contains the final method, the paper ablations and selected development utilities; non-paper variants
(e.g. the other CARE variants in `track2/care.py`) are disabled by default and not used by the reported model.

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
switching. The learned fusion is strongly stereo-dominant: measured fusion weights stay ≥ 0.991 on every validation
pixel (`results/fusion_weights.csv`, `results/mono_stereo_fused_error.csv`).

## 2. Layout
| path | content |
|---|---|
| `track2/` | library: `croco_model.py` (wrapper, tiled inference), `care.py` (correspondence dropout, random-key control), `mono.py` (mono head + fuser), `s2aug.py`, `stereo_corrmask.py`, `twin_data.py`, `losses.py`, `backbones.py`, `robust20.py` (local corruption suite), `val_suite.py` (metrics), `config.py`, `paths.py` |
| `scripts/train_track2.py` | training entry point (writes `contract.json` with config, seed, source and checkpoint hashes) |
| `scripts/eval_suite.py` | evaluation on the frozen local suite (`--suite robust20`) |
| `scripts/build_robust20_val.py` | builds the frozen 20-corruption validation cache (severity 3) |
| `scripts/run_train_eval.sh` | `<gpu> <config> <name>`: train + robust20 evaluation of the final checkpoint |
| `scripts/make_submission.py`, `run_submission.sh` | test predictions (21 conditions × 1000 frames × 2 views) → official subsampling binaries → the two upload files |
| `scripts/final_seed_summary.py`, `final_targeted_summary.py` | seed / ablation tables of the paper |
| `scripts/final_analysis.py`, `analyze_evidence_free.py`, `figures/` | regional (sky / textureless) analysis, fusion weights, latency, qualitative figures |
| `configs/track2/` | the final configs and their inheritance chain (29 files) |
| `splits/` | scene-disjoint split: 4054 train pairs, 49 validation frames (28 primary: scenes 0038/0039/0045/0047), `track2_train_all.txt` (4103 pairs, submission model), corruption seeds |
| `tests/` | CPU test suite (`python3 -m pytest -q tests`); `gpu_croco_tiling_parity.py` = optional GPU parity check of the tiled inference |
| `results/` | every table of the paper as csv/md (`e0_baseline_summary.csv` = clean-only baseline row), regional / fusion-weight / latency analyses, appendix figure, submission bundle hashes |
| `third_party/croco-source.json` | pinned CroCo commit |

Configs: `track2_croco_x7_x6b_mono` = full model (local protocol, seed 2026); `track2_croco_abl_seed2/3` = seeds 2027/2028;
`track2_croco_final_submit` = leaderboard model (all 4103 pairs, init = official Spring-finetuned CroCo-Stereo);
`track2_croco_abl_*` = ablations (each differs from the full config only in the ablated key — enforced by a unit test);
`track2_croco_abl_randkey_s1/s2` = budget-matched random-key dropout control; `track2_e0_croco_clean` = clean-only baseline E0 / teacher.

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
# clean-only baseline E0 = frozen teacher of all robust runs
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_track2.py --config track2_e0_croco_clean --out outputs/track2/e0
mkdir -p checkpoints/track2 && cp outputs/track2/e0/final.pt checkpoints/track2/croco_clean_teacher.pt
# full model / seeds / ablations: train + robust20 evaluation -> outputs/track2/final_runs/<name>/{train,eval}
scripts/run_train_eval.sh 0 track2_croco_x7_x6b_mono full_seed2026
scripts/run_train_eval.sh 1 track2_croco_abl_seed2   full_seed2027
scripts/run_train_eval.sh 2 track2_croco_abl_no_dropout no_dropout_seed2026      # … any track2_croco_abl_* config
# leaderboard model and submission files
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_track2.py --config track2_croco_final_submit --out outputs/track2/final_submit
scripts/run_submission.sh outputs/track2/final_submit/final.pt final_x7 "0 1 2 3"
#   -> submissions/track2/final_x7/bundle/clean/disp1_submission.hdf5   (~141 MB)
#      submissions/track2/final_x7/bundle/robust/disp1_robustness.hdf5  (~161 MB, 21 groups)
```
Upload only those two files (raw predictions are ~250 GB and can be deleted after bundling). `train_track2.py` refuses
to write into an existing `--out` folder.

**Tables and figures.** The committed files in `results/` are the outputs of the runs above. The summary scripts
(`scripts/final_seed_summary.py`, `final_targeted_summary.py`) read the evaluation folders listed in their `RUNS`
tables (our original run names under `outputs/track2/`); adapt `RUNS` to your folder names. The regional /
fusion-weight / latency analyses and the qualitative figures need model predictions that are not committed:

    checkpoints (E0 + full model, paths in MODELS of scripts/final_analysis.py)
      -> python3 scripts/final_analysis.py predict      (prediction cache outputs/track2/final_analysis/cache, GPU)
      -> python3 scripts/final_analysis.py stats | figures | latency
      -> python3 scripts/figures/qualitative_robustness_appendix.py figure

The final appendix figure itself is committed in `results/figures/`.

## 5. Results
Official [Spring stereo benchmark](https://spring-benchmark.org/stereo) entry **RoCo-19-CorrDrop-Stereo**
(rank 1 as of 2026-09-23; model `track2_croco_final_submit`):

| Abs | 1px | D1 |
|---|---|---|
| 0.457 | 7.095 | 2.615 |

Robustness columns of the same leaderboard:

| ΔAbs | Δ1px | ΔD1 |
|---|---|---|
| 1.441 | 18.088 | 1.909 |

See the leaderboard for the per-region breakdown and metric definitions.

Local protocol (28 primary validation frames, 20 corruptions, mean ΔAbs; mean ± sample sd over seeds):

| variant | seeds | ΔAbs | clean Abs |
|---|---|---|---|
| clean-only baseline E0 | 1 | 2.195 | 0.281 |
| **full** | 3 | **0.670 ± 0.107** | 0.287 ± 0.001 |
| no correspondence dropout | 2 | 0.866 ± 0.039 | 0.286 |
| random-key dropout (budget-matched control) | 2 | 1.223 ± 0.229 | 0.285 |
| no L_delta | 2 | 0.965 ± 0.022 | 0.280 |
| image-plane mask transport | 2 | 0.869 ± 0.016 | 0.289 |
| no teacher / anchor | 2 | 0.829 ± 0.129 | 0.309 |
| no mono head | 2 | 0.727 ± 0.055 | 0.284 |
| no sensor overlay | 2 | 0.716 ± 0.082 | 0.289 |

With 2–3 seeds per variant these are descriptive statistics; no significance is claimed. Details, per-corruption values
and matched-seed comparisons: `results/`.

## 6. Declared data, checkpoints and compute
Training data: Spring train split only (scene-disjoint local split for the paper, all 4103 pairs for the leaderboard
model). Initialisation: official CroCo-Stereo weights (public pre-Spring weights for local runs, the official
Spring-finetuned weights for the leaderboard model, which also serve as its frozen teacher). No RobustSpring test image,
label or leaderboard feedback was used for training or model selection; the same checkpoint and settings produced both
uploads (`results/BUNDLE_CONTRACT.json`: checkpoint sha256 3e579178…1d42). The local corruption suite is a
re-implementation (ImageNet-C style + depth-aware fog), not the RobustSpring generator. No git on the training host: every
run records the sha256 of its sources in `contract.json`. Hardware: NVIDIA L40S (48 GB), ≈ 5 GPU-hours per training run.

## 7. Licences
Code written for this repository: CC BY-NC-SA 4.0 (`LICENSE`), matching CroCo / CroCo-Stereo, which it builds on and
whose tiled inference and loss it re-implements. Non-commercial use only. External code, checkpoints and datasets
are not redistributed; their terms are listed in `THIRD_PARTY_NOTICES.md`.

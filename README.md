# RoCo-19 – KoreaU_DLmath · RoCo-Spring Track 2 (stereo) · reproducibility package

Code for the leaderboard entry **RoCo-19-CorrDrop-Stereo** and for every table / figure of the workshop paper.
Only the final method is included; optional variants that are still selectable in the code are off by default.

## 1. Method in one paragraph
CroCo-Stereo v2 (ViT-L encoder, ViT-B decoder, DPT head; 437.4 M parameters) is fine-tuned on Spring with a
clean/corrupted **twin** forward. The corrupted branch sees S2Aug (7 generic corruption families, stereo-consistent,
occluder masks transported to the right view with a disparity z-buffer), CorrMask (teacher-confidence-guided box
occlusions), sensor overlays, and **correspondence dropout**: for a random rectangular region of left queries, the right-view
keys within ±1 token row (the epipolar band) are masked in all 12 decoder cross-attention layers, so the prediction must
come from context. Losses: `L_clean + 0.25 L_corr + 1.0 L_delta + 0.1 L_anchor` (twin consistency and a frozen
clean-finetuned teacher). A small monocular DPT head + per-pixel fuser (+20.4 M parameters) is attached to the detached left
encoder tokens; measured fusion weights stay ≥ 0.991, i.e. inference is effectively the stereo branch (see
`results/fusion_weights.csv`, `results/mono_stereo_fused_error.csv`). Fog, frost, speckle noise, zoom blur and elastic transform are never used in training.
Inference is the unchanged network: tiles 352×704, overlap 0.7, fp32, no test-time adaptation.

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
| `tests/` | 59 CPU unit tests (`python3 -m pytest -q tests`) |
| `results/` | every table of the paper as csv/md, regional / fusion-weight / latency analyses, appendix figure, submission bundle hashes |
| `third_party/croco-source.json` | pinned CroCo commit |

Configs: `track2_croco_x7_x6b_mono` = full model (local protocol, seed 2026); `track2_croco_abl_seed2/3` = seeds 2027/2028;
`track2_croco_final_submit` = leaderboard model (all 4103 pairs, init = official Spring-finetuned CroCo-Stereo);
`track2_croco_abl_*` = ablations (each differs from the full config only in the ablated key — enforced by a unit test);
`track2_croco_abl_randkey_s1/s2` = budget-matched random-key dropout control; `track2_e0_croco_clean` = clean-only baseline E0 / teacher.

## 3. Setup
Python 3.11, PyTorch 2.7.1 + CUDA 11.8, numpy 2.2, h5py 3.16, timm 1.0, einops 0.8, scikit-image 0.26, scipy, pillow, pyyaml,
matplotlib. One 48 GB GPU is enough (training peak 17 GB with `micro_batch: 1`; ≈ 5 h per run on an L40S).

```bash
git clone https://github.com/naver/croco third_party/croco && git -C third_party/croco checkout 5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb
(cd third_party/croco/models/curope && python3 setup.py build_ext --inplace)          # cuRoPE kernels
mkdir -p checkpoints/croco_official                                                   # official CroCo-Stereo weights (see CroCo README)
#   crocostereo.pth                  sha256 8b48935772c98714db1a18773d687da273d3ab721410cc5e398451a0dac8982f
#   crocostereo_finetune_spring.pth  sha256 123273a722a58134c6816efe5122c427815c92f82896be4d484e6d21b52da575
# defaults (relative to this folder): data/spring, data/, third_party/subsampling_tools, third_party/croco
export ROCO_SPRING=/path/to/spring            # contains train/ and test/
export ROCO_DATA=/path/to/roco                # spring/ + RobustSpring test corruptions (submission only)
export ROCO_SUBSAMPLING_TOOLS=/path/to/subsampling_tools   # official Spring binaries (submission only)
# optional: ROCO_TRACK2_WORK=<dir holding checkpoints/, outputs/, third_party/croco>  (default: this folder)
```
All paths are defined in `track2/paths.py`. `ROCO_TRACK3_ROOT` is only needed for the optional DEFOM backbone and the
Q4Aug source-parity test (skipped / irrelevant for the CroCo results).

## 4. Reproduce
```bash
python3 -m pytest -q tests
python3 scripts/build_robust20_val.py                                            # frozen local robust20 cache (49 frames × 20)
# clean-only baseline E0 = frozen teacher  ->  copy its final.pt to checkpoints/track2/croco_clean_teacher.pt
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_track2.py --config track2_e0_croco_clean --out outputs/track2/e0
# full model / seeds / ablations (train + robust20 evaluation)
scripts/run_train_eval.sh 0 track2_croco_x7_x6b_mono full_seed2026
scripts/run_train_eval.sh 1 track2_croco_abl_seed2   full_seed2027
scripts/run_train_eval.sh 2 track2_croco_abl_no_dropout no_dropout_seed2026      # … any track2_croco_abl_* config
# leaderboard model and submission files
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_track2.py --config track2_croco_final_submit --out outputs/track2/final_submit
scripts/run_submission.sh outputs/track2/final_submit/final.pt final_x7 "0 1 2 3"
#   -> submissions/track2/final_x7/bundle/clean/disp1_submission.hdf5   (~141 MB)
#      submissions/track2/final_x7/bundle/robust/disp1_robustness.hdf5  (~161 MB, 21 groups)
```
Upload only those two files (raw predictions are ~250 GB and can be deleted after bundling). The summary scripts expect
the run folders listed at the top of `scripts/final_seed_summary.py` / `final_targeted_summary.py`; adapt the `RUNS`
table to your folder names.

## 5. Results
Official leaderboard (RoCo-19-CorrDrop-Stereo): clean Abs 0.457 / 1px 7.095 / D1 2.615; robustness ΔAbs 1.441 / Δ1px 18.088 /
ΔD1 1.909. Local protocol (28 primary frames, 20 corruptions, mean ΔAbs; mean ± sample sd over seeds):

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
CroCo / CroCo-Stereo: CC BY-NC-SA 4.0 (NAVER) — not redistributed here, fetched by the setup step. Spring / RobustSpring:
see the dataset terms. `track2/q4_primitives.py` contains augmentation primitives from our Track 3 entry.

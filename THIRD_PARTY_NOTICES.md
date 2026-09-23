# Third-party notices

## 1. Code written for this repository
Everything under `track2/`, `scripts/`, `configs/`, `tests/`, `splits/` and `results/` was written by the authors
(RoCo-19 – KoreaU_DLmath) and is released under CC BY-NC-SA 4.0 (`LICENSE`). Parts of `track2/croco_model.py`
(tiled inference) and `track2/losses.py` (Laplacian loss wrapper) re-implement the corresponding CroCo-Stereo
functions (`stereoflow/engine.py`, `stereoflow/criterion.py`); this is why the CroCo licence is adopted.

## 2. External code (not redistributed; fetched by the user)
| component | source | licence (verified from the pinned checkout) | used for |
|---|---|---|---|
| CroCo / CroCo-Stereo | https://github.com/naver/croco @ `5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb` (`third_party/croco-source.json`) | CC BY-NC-SA 4.0, © NAVER Corporation (`LICENSE` of that commit). Its `NOTICE` lists sub-components under their own terms: facebookresearch/mae (CC BY-NC 4.0) and rwightman/pytorch-image-models (Apache 2.0) | network definition, DPT head, cuRoPE, stereo augmentor |
| `imagecorruptions` 1.1.2 | PyPI | not verified here — see the package metadata / repository | only the frost texture images are read (`track2/robust20.py`); its code is not imported |
| Python packages in `requirements.txt`, PyTorch | PyPI / pytorch.org | their respective licences | runtime |
| DEFOM-Stereo, FoundationStereo (optional) | upstream repositories | not verified here; not needed for any reported CroCo result | optional backbones in `track2/backbones.py`, `scripts/eval_suite.py` |

## 3. Vendored code
`track2/q4_primitives.py` contains augmentation primitives copied from the authors' own RoCo Track 3 code (same
authors; released here under this repository's licence). No third-party source files are vendored.

## 4. Checkpoints and data (obtain separately)
| item | source | terms |
|---|---|---|
| `crocostereo.pth`, `crocostereo_finetune_spring.pth` | NAVER LABS Europe (CroCo `stereoflow/download_model.sh`) | CroCo licence (CC BY-NC-SA 4.0) and CroCo `NOTICE_CHECKPOINTS` (datasets used for pre-training carry additional terms) |
| Spring dataset | https://spring-benchmark.org | the website states that the Spring movie assets (Blender Foundation) are CC BY 4.0; the dataset's own terms of use must be checked with the providers |
| RobustSpring test corruptions, official subsampling binaries | RoCo challenge / RobustSpring organisers | provider terms (not verified here) |
| Checkpoints trained with this code | not distributed | derived from CroCo-Stereo weights, hence subject to the CroCo licence and its checkpoint notices |

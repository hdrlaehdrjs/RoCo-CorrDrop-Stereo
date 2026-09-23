# Third-party notices

## 1. Code written for this repository
Everything under `track2/`, `scripts/`, `configs/`, `tests/`, `splits/` and `results/` was written by the authors
(RoCo-19 – KoreaU_DLmath) and is released under CC BY-NC-SA 4.0 (`LICENSE`). Parts of `track2/croco_model.py`
(tiled inference) and `track2/losses.py` (Laplacian loss wrapper) re-implement the corresponding CroCo-Stereo
functions (`stereoflow/engine.py`, `stereoflow/criterion.py`); this is why the CroCo licence is adopted.
`track2/robust20.py` re-implements, with documented modifications, the ImageNet-C corruption functions of
hendrycks/robustness and bethgelab/imagecorruptions (both Apache-2.0, https://www.apache.org/licenses/LICENSE-2.0; attribution and a description of the
changes are kept in the module docstring).

## 2. External code (not redistributed; fetched by the user)
| component | source | licence (verified) | used for |
|---|---|---|---|
| CroCo / CroCo-Stereo | https://github.com/naver/croco @ `5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb` (`third_party/croco-source.json`) | CC BY-NC-SA 4.0, © NAVER Corporation (`LICENSE` of that commit). Its `NOTICE` lists sub-components under their own terms: facebookresearch/mae (CC BY-NC 4.0) and rwightman/pytorch-image-models (Apache 2.0) | network definition, DPT head, cuRoPE, stereo augmentor |
| `imagecorruptions` 1.1.2 | PyPI, https://github.com/bethgelab/imagecorruptions | Apache-2.0 (LICENSE file shipped with the 1.1.2 wheel; GitHub licence metadata). Extends hendrycks/robustness (ImageNet-C), also Apache-2.0 | only the frost texture images are read at run time (`track2/robust20.py`); not redistributed |
| Python packages in `requirements.txt`, PyTorch | PyPI / pytorch.org | their respective licences | runtime |
| DEFOM-Stereo (optional) | https://github.com/Insta360-Research-Team/DEFOM-Stereo | MIT (LICENSE of the checkout used) | optional backbone in `track2/backbones.py`; not needed for the CroCo results |
| FoundationStereo (optional) | https://github.com/NVlabs/FoundationStereo | NVIDIA source-code licence, non-commercial / research use only (upstream LICENSE) | optional zero-shot reference in `scripts/eval_suite.py`; not needed for the CroCo results |

## 3. Vendored code
`track2/q4_primitives.py` contains augmentation primitives copied from the authors' own RoCo Track 3 code (same
authors; released here under this repository's licence). No third-party source files are vendored.

## 4. Checkpoints and data (obtain separately)
| item | source | terms |
|---|---|---|
| `crocostereo.pth`, `crocostereo_finetune_spring.pth` | NAVER LABS Europe (CroCo `stereoflow/download_model.sh`) | CroCo licence (CC BY-NC-SA 4.0) and CroCo `NOTICE_CHECKPOINTS` (datasets used for pre-training carry additional terms) |
| Spring dataset | https://spring-benchmark.org, DaRUS doi:10.18419/darus-3376 | CC BY 4.0 (DaRUS record licence; Spring movie assets by Blender Foundation, CC BY 4.0) |
| RobustSpring corruptions | DaRUS doi:10.18419/DARUS-5047 | CC BY 4.0 (DaRUS record licence) |
| official Spring subsampling binaries | distributed by the RoCo challenge organisers | no licence file is shipped with the binaries; used unmodified to create the submission files and not redistributed |
| Checkpoints trained with this code | not distributed | derived from CroCo-Stereo weights, hence subject to the CroCo licence and its checkpoint notices |

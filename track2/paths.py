"""Filesystem anchors. Everything machine-specific can be overridden with environment variables:
  ROCO_TRACK2_WORK  working directory holding checkpoints/, outputs/ and third_party/croco (default: this code folder)
  ROCO_SPRING       Spring dataset root (contains train/ and test/)
  ROCO_DATA         folder holding spring/ and the RobustSpring test corruptions (submission only)
  ROCO_CROCO_ROOT   checkout of github.com/naver/croco @ 5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb (cuRoPE compiled)
  ROCO_SUBSAMPLING_TOOLS  folder with the official Spring binaries disp1_subsampling / disp1_robust_subsampling
  ROCO_TRACK3_ROOT  only for the optional DEFOM backbone and the Q4Aug source-parity test
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("ROCO_TRACK2_WORK", ROOT))
TRACK3_ROOT = Path(os.environ.get("ROCO_TRACK3_ROOT", "/DLMATH/KDG/Image/Roco"))
CROCO_ROOT = Path(os.environ.get("ROCO_CROCO_ROOT", WORK / "third_party/croco"))
SPRING = Path(os.environ.get("ROCO_SPRING", "/DLMATH/Data/roco/spring"))
OFFICIAL_CKPT = WORK / "checkpoints/croco_official"
TRACK2_CKPT = WORK / "checkpoints/track2"
OUTPUTS = WORK / "outputs/track2"
SPLITS = ROOT / "splits"

# B_S and B_R_proxy of the official Track 2 stereo scoring (given by the organisers).
B_S = 3.4545
B_R_PROXY = 18.4505


def add_croco_to_path() -> None:
    p = str(CROCO_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

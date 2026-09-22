"""Spring stereo readers and the fixed Track 2 scene-level split.

The split REUSES Track 3's scene-disjoint holdout (Roco/configs/spring_holdout.json):
validation scenes 0038 0039 0041 0043 0044 0045 0047; validation frames are the same
49 frames Track 3 selects (7 per scene, np.linspace over the flow_FW_left listing,
identical to track3_defom_dpflow.validation_pairs). Stereo training uses every left frame
of the remaining scenes (4054 frames).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .paths import SPRING, SPLITS, TRACK3_ROOT

TRACK3_HOLDOUT = SPLITS / "spring_holdout.json"   # copy of the Track 3 hold-out definition (scene-disjoint split)


@dataclass(frozen=True)
class StereoFrame:
    scene: str
    frame: int

    @property
    def key(self) -> str:
        return f"{self.scene}:{self.frame:04d}"


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()


def read_disp_full(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["disparity"][()].astype(np.float32)


def read_frame(fr: StereoFrame, root: Path = SPRING, full_gt: bool = False) -> dict:
    s = root / "train" / fr.scene
    d = read_disp_full(s / "disp1_left" / f"disp1_left_{fr.frame:04d}.dsp5")
    out = dict(left=read_rgb(s / "frame_left" / f"frame_left_{fr.frame:04d}.png"),
               right=read_rgb(s / "frame_right" / f"frame_right_{fr.frame:04d}.png"),
               disp=np.ascontiguousarray(d[::2, ::2]))  # Track 3 convention: strided to image grid
    if full_gt:
        out["disp_full"] = d
    return out


def holdout_config() -> dict:
    return json.loads(TRACK3_HOLDOUT.read_text())


def validation_frames(root: Path = SPRING) -> list[StereoFrame]:
    split = holdout_config()
    frames = []
    for scene in split["validation_scenes"]:
        flows = sorted((root / "train" / scene / "flow_FW_left").glob("*.flo5"))
        idx = np.linspace(0, len(flows) - 1, min(split["samples_per_scene"], len(flows)), dtype=int)
        frames += [StereoFrame(scene, int(flows[i].stem.rsplit("_", 1)[-1])) for i in idx]
    return frames


def train_frames(root: Path = SPRING) -> list[StereoFrame]:
    val_scenes = set(holdout_config()["validation_scenes"])
    frames = []
    for scene in sorted((root / "train").iterdir()):
        if not scene.is_dir() or scene.name in val_scenes:
            continue
        for p in sorted((scene / "frame_left").glob("*.png")):
            f = int(p.stem.rsplit("_", 1)[-1])
            if (scene / "frame_right" / f"frame_right_{f:04d}.png").exists() and \
               (scene / "disp1_left" / f"disp1_left_{f:04d}.dsp5").exists():
                frames.append(StereoFrame(scene.name, f))
    return frames


def write_split_manifests() -> dict:
    SPLITS.mkdir(parents=True, exist_ok=True)
    tr, va = train_frames(), validation_frames()
    assert not ({f.scene for f in tr} & {f.scene for f in va})
    (SPLITS / "track2_train.txt").write_text("\n".join(f.key for f in tr) + "\n")
    (SPLITS / "track2_val.txt").write_text("\n".join(f.key for f in va) + "\n")
    return dict(train=len(tr), val=len(va), val_scenes=holdout_config()["validation_scenes"])


def load_split(name: str) -> list[StereoFrame]:
    lines = (SPLITS / f"track2_{name}.txt").read_text().split()
    return [StereoFrame(l.split(":")[0], int(l.split(":")[1])) for l in lines]

"""Twin (clean, S2Aug-corrupted) stereo training samples with a step-indexed deterministic stream."""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .s2aug import S2AugConfig, resize_crop, s2aug
from .spring_data import read_frame


class StepSampler(Sampler):
    """Yields (step, slot, frame_index); epoch-wise seeded permutation; resumable at any step."""

    def __init__(self, n_frames, batch_size, total_steps, seed, start_step=0):
        self.n, self.bs, self.total, self.seed, self.start = n_frames, batch_size, total_steps, seed, start_step

    def order(self, epoch):
        return torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + 1000003 * epoch)).tolist()

    def __iter__(self):
        cache = {}
        for step in range(self.start, self.total):
            for slot in range(self.bs):
                k = step * self.bs + slot
                ep = k // self.n
                if ep not in cache:
                    cache = {ep: self.order(ep)}
                yield (step, slot, cache[ep][k % self.n])

    def __len__(self):
        return (self.total - self.start) * self.bs


class TwinStereoDataset(Dataset):
    def __init__(self, frames, crop, s2cfg: S2AugConfig, total_steps: int, seed: int, geometry: str = "track3"):
        self.frames, self.crop, self.cfg, self.total, self.seed, self.geometry = frames, tuple(crop), s2cfg, total_steps, seed, geometry

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, key):
        step, slot, fi = key
        cv2.setNumThreads(0)
        s = self.seed + 7919 * step + slot
        rng, nr = random.Random(s), np.random.default_rng(s)
        fr = self.frames[fi]
        it = read_frame(fr)
        if self.geometry == "croco_official":  # upstream StereoAugmentor (scale-x, crop, vflip, right jitter, asym color)
            from stereoflow.augmentor import StereoAugmentor
            np.random.seed(s % 2**32); random.seed(s)
            L, R, D = StereoAugmentor(self.crop)(it["left"], it["right"], it["disp"], "Spring")
            L, R, D = (np.ascontiguousarray(x) for x in (L, R, D))
            rng = random.Random(s + 1)
        elif self.geometry == "crop_only":  # plain random crop, as in Track 3 finetune_defom_spring
            h, w = it["disp"].shape
            y, x = rng.randrange(h - self.crop[0] + 1), rng.randrange(w - self.crop[1] + 1)
            sl = (slice(y, y + self.crop[0]), slice(x, x + self.crop[1]))
            L, R, D = (np.ascontiguousarray(it[k][sl]) for k in ("left", "right", "disp"))
        else:
            L, R, D = resize_crop(it["left"], it["right"], it["disp"], self.crop, rng)
        if type(self.cfg).__name__ == "CVCAugConfig":
            from .cvcaug import cvcaug
            Lx, Rx, info = cvcaug(L, R, D, rng, nr, self.cfg, progress=step / max(1, self.total))
            info.setdefault("severity", max(info["sev_L"], info["sev_R"]))
            info.setdefault("mask_event", 0)
            info.setdefault("mask_world", 0)
        else:
            Lx, Rx, info = s2aug(L, R, D, rng, nr, self.cfg, progress=step / max(1, self.total))
            info.update(mode="s2aug", sev_L=info["severity"], sev_R=info["severity"], mask_area=0.0, recovery=-1)
        t = lambda a: torch.from_numpy(a).permute(2, 0, 1).float()
        return dict(clean_left=t(L), clean_right=t(R), corr_left=t(Lx), corr_right=t(Rx),
                    disp=torch.from_numpy(D)[None], frame=fr.key, step=step,
                    aug=info["aug"], severity=info["severity"], mask_event=info["mask_event"],
                    mask_world=info["mask_world"], mask_seed=info["mask_seed"], mode=info["mode"],
                    sev_L=float(info["sev_L"]), sev_R=float(info["sev_R"]), mask_area=float(info["mask_area"]),
                    recovery=int(info["recovery"]), overlay=str(info.get("overlay", "none")),
                    erase=int(info.get("erase", 0)), erase_area=float(info.get("erase_area", 0.0)))

"""CVCAug: Cross-View Correspondence Augmentation (CARE-Stereo data level).

Built only from already tested pieces: Track 3 Q4Aug primitives (q4_primitives: _photo_params/_photo, _degrade,
weather_mask, _overlay) and the S2Aug z-buffer left->right transport (s2aug.transport_np). Geometry and the GT
disparity are never modified (appearance / occluder overlays only).

Modes (one per corrupted sample, after the p_aug gate):
  shared      : same photometric params + same family/severity on both views; weather occluders transported L->R
  asymmetric  : one randomly chosen view gets a corruption at severity s; the other view is either clean
                (p_asym_clean) or gets the same family at s * U(0, weak_ratio); independent photometric params
  corr_dropout: valid-disparity-aware random box M_L (Track 3 box size), M_R = W_stereo(M_L, d_gt)
                recovery   (p_recovery): destroy M_L only -> the match is still visible in the right view
                shared_hard(1 - p_recovery): destroy M_L and M_R
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from . import q4_primitives as q4
from .s2aug import S2AugConfig, transport_np

FAMILIES = ("photo", "blur", "noise", "jpeg", "pixelate", "rain", "snow", "spatter")
WEATHER = ("rain", "snow", "spatter")


@dataclass
class CVCAugConfig:
    p_aug: float = 0.8
    p_shared: float = 0.35
    p_asym: float = 0.35
    p_corr_dropout: float = 0.30
    p_recovery: float = 0.7
    p_asym_clean: float = 0.5
    weak_ratio: float = 0.5
    severity: tuple = (0.15, 0.6)          # mild-to-medium (S2Aug used 0.15-0.75)
    families: tuple = FAMILIES
    n_boxes: tuple = (1, 2)
    box_frac_h: tuple = (1 / 12, 1 / 4)    # Track 3 CorrMask box ranges
    box_frac_w: tuple = (1 / 12, 1 / 4)
    mask_sampling: str = "random_valid"   # random_valid (gt_attention_reliable is not implemented yet)
    warmup_frac: float = 0.1              # S2Aug-style curriculum on p_aug and max severity
    p_aug_start: float = 0.4
    severity_hi_start: float = 0.35

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


def _corrupt(img, kind, sev, photo, rng, nr, mask=None):
    """Apply one family to one view. `mask` (weather occluder) overrides a freshly drawn one."""
    out = q4._photo(img, photo) if photo is not None else img
    if kind == "photo":
        return out
    if kind in WEATHER:
        m = mask if mask is not None else q4.weather_mask(img.shape[:2], kind, sev, rng, nr)
        return q4._overlay(out, m, nr, white=True)
    return q4._degrade(out, kind, sev, nr)


def _valid_box(disp, rng, cfg):
    h, w = disp.shape
    valid = np.isfinite(disp) & (disp >= 0)
    xs = np.arange(w)[None, :].repeat(h, 0)
    valid &= (xs - np.nan_to_num(disp)) >= 0          # correspondence inside the right image
    ys, xs_ = np.nonzero(valid)
    if len(ys) == 0:
        return None
    i = rng.randrange(len(ys))
    y, x = int(ys[i]), int(xs_[i])
    mh = rng.randint(max(16, int(h * cfg.box_frac_h[0])), max(24, int(h * cfg.box_frac_h[1])))
    mw = rng.randint(max(24, int(w * cfg.box_frac_w[0])), max(32, int(w * cfg.box_frac_w[1])))
    m = np.zeros((h, w), np.float32)
    m[max(0, y - mh // 2):min(h, y + mh // 2), max(0, x - mw // 2):min(w, x + mw // 2)] = 1
    return cv2.GaussianBlur(m, (0, 0), 1.2)


def cvcaug(left, right, disp, rng: random.Random, nr: np.random.Generator, cfg: CVCAugConfig, progress: float = 1.0):
    """Returns (left~, right~, info). `disp` is read-only; images keep their size and geometry."""
    L, R = left.copy(), right.copy()
    if cfg.warmup_frac > 0 and progress < cfg.warmup_frac:
        t = progress / cfg.warmup_frac
        p_aug = cfg.p_aug_start + t * (cfg.p_aug - cfg.p_aug_start)
        sev_hi = cfg.severity_hi_start + t * (cfg.severity[1] - cfg.severity_hi_start)
    else:
        p_aug, sev_hi = cfg.p_aug, cfg.severity[1]
    info = dict(aug="clean", mode="clean", kind="none", sev_L=0.0, sev_R=0.0, mask_area=0.0, recovery=-1)
    if rng.random() >= p_aug:
        info["mask_seed"] = rng.randrange(2**31)
        return L, R, info
    u = rng.random() * (cfg.p_shared + cfg.p_asym + cfg.p_corr_dropout)
    lo = cfg.severity[0]
    s2t = S2AugConfig(transport="zbuffer")
    if u < cfg.p_shared:
        kind = rng.choice(tuple(cfg.families))
        sev = rng.uniform(lo, max(lo, sev_hi))
        photo = q4._photo_params(rng)
        mL = mR = None
        if kind in WEATHER:
            mL = q4.weather_mask(disp.shape, kind, sev, rng, nr)
            mR = transport_np(mL, disp, s2t)
        L = _corrupt(L, kind, sev, photo, rng, nr, mL)
        R = _corrupt(R, kind, sev, photo, rng, nr, mR)
        info.update(mode="shared", kind=kind, sev_L=sev, sev_R=sev)
    elif u < cfg.p_shared + cfg.p_asym:
        kind = rng.choice(tuple(cfg.families))
        sev = rng.uniform(lo, max(lo, sev_hi))
        strong_left = rng.random() < 0.5
        weak = 0.0 if rng.random() < cfg.p_asym_clean else sev * rng.uniform(0, cfg.weak_ratio)
        sL, sR = (sev, weak) if strong_left else (weak, sev)
        pL, pR = q4._photo_params(rng), q4._photo_params(rng)
        if sL > 0:
            L = _corrupt(L, kind, sL, pL, rng, nr)
        if sR > 0:
            R = _corrupt(R, kind, sR, pR, rng, nr)
        info.update(mode="asymmetric", kind=kind, sev_L=sL, sev_R=sR)
    else:
        n = rng.randint(*cfg.n_boxes)
        mL = np.zeros(disp.shape, np.float32)
        for _ in range(n):
            b = _valid_box(disp, rng, cfg)
            if b is not None:
                mL = np.maximum(mL, b)
        mR = transport_np(mL, disp, s2t)
        recovery = rng.random() < cfg.p_recovery
        L = q4._overlay(L, mL, nr, white=False)
        if not recovery:
            R = q4._overlay(R, mR, nr, white=False)
        info.update(mode="corr_dropout", kind="recovery" if recovery else "shared_hard", recovery=int(recovery),
                    mask_area=float((mL > .5).mean()), sev_L=1.0, sev_R=0.0 if recovery else 1.0)
    info["aug"] = f"cvc:{info['mode']}:{info['kind']}"
    info["mask_seed"] = rng.randrange(2**31)
    return L, R, info

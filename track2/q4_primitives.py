"""Q4Aug / CorrMask primitives VENDORED VERBATIM from Track 3.

Source: /DLMATH/KDG/Image/Roco/scripts/corresguard_core.py (functions _photo_params,
_photo, _degrade, weather_mask, forward_splat_mask, _overlay, _priority_mask) and
/DLMATH/KDG/Image/Roco/scripts/paper_ablation_aug.py (random_mask).

Why vendored instead of imported: importing corresguard_core executes
`import track3_defom_dpflow`, which imports DEFOM-Stereo and ptlflow/DPFlow. Track 2
S2Aug must have no flow dependency, and the Track 3 tree must not be edited.
tests/test_q4_parity.py checks bit-identical outputs against the Track 3 originals
and pins the Track 3 source hashes, so the Track 3 pipeline is provably unchanged.

Only `_priority_mask` needs flow; the stereo-only version lives in stereo_corrmask.py.
"""
from __future__ import annotations

import random

import cv2
import numpy as np

TRACK3_SOURCE_SHA256 = {
    # filled/verified by tests/test_q4_parity.py
    "scripts/corresguard_core.py": None,
    "scripts/visibility_consistency_core.py": None,
    "scripts/paper_ablation_aug.py": None,
}


def _photo_params(rng: random.Random) -> tuple[float, float, float, float, bool]:
    return (rng.uniform(.75, 1.25), rng.uniform(.75, 1.25),
            rng.uniform(.7, 1.3), rng.uniform(.8, 1.2), rng.random() < .04)


def _photo(image: np.ndarray, p: tuple[float, float, float, float, bool]) -> np.ndarray:
    brightness, contrast, saturation, gamma, gray_flag = p
    x = image.astype(np.float32) / 255
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * contrast + mean
    x *= brightness
    gray = x.mean(axis=2, keepdims=True)
    x = gray + saturation * (x - gray)
    x = np.clip(x, 0, 1) ** gamma
    if gray_flag:
        x = np.repeat(x.mean(axis=2, keepdims=True), 3, axis=2)
    return np.clip(255 * x, 0, 255).astype(np.uint8)


def _degrade(image: np.ndarray, kind: str, severity: float, np_rng: np.random.Generator) -> np.ndarray:
    if kind == "blur":
        return cv2.GaussianBlur(image, (0, 0), .25 + 1.6 * severity)
    if kind == "noise":
        noise = np_rng.normal(0, 2 + 11 * severity, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        quality = int(92 - 57 * severity)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind == "pixelate":
        h, w = image.shape[:2]
        factor = .9 - .5 * severity
        small = cv2.resize(image, (max(2, int(w * factor)), max(2, int(h * factor))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return image


def weather_mask(shape: tuple[int, int], kind: str, severity: float,
                 rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    if kind == "rain":
        for _ in range(int((70 + 180 * severity) * h * w / (256 * 512))):
            x, y, length = rng.randrange(w), rng.randrange(h), rng.randint(8, 28)
            cv2.line(mask, (x, y), (min(w - 1, x + length // 4), min(h - 1, y + length)),
                     rng.uniform(.5, 1), rng.randint(1, 2))
        mask = cv2.GaussianBlur(mask, (3, 3), .7)
    elif kind == "snow":
        count = int((150 + 500 * severity) * h * w / (256 * 512))
        ys, xs = np_rng.integers(0, h, count), np_rng.integers(0, w, count)
        mask[ys, xs] = np_rng.uniform(.6, 1, count)
        mask = np.clip(cv2.GaussianBlur(mask, (0, 0), .7 + severity) * 3, 0, 1)
    else:
        for _ in range(int(8 + 28 * severity)):
            cv2.circle(mask, (rng.randrange(w), rng.randrange(h)), rng.randint(5, 24),
                       rng.uniform(.35, .9), -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
    return np.clip(mask, 0, 1)


def forward_splat_mask(mask: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Nearest forward splat with max aggregation, suitable for occluder masks."""
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    tx = np.rint(xx + np.nan_to_num(dx)).astype(np.int64)
    ty = np.rint(yy + np.nan_to_num(dy)).astype(np.int64)
    valid = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h) & (mask > 0)
    result = np.zeros(h * w, np.float32)
    np.maximum.at(result, (ty[valid] * w + tx[valid]), mask[valid])
    return cv2.dilate(result.reshape(h, w), np.ones((3, 3), np.uint8))


def _overlay(image: np.ndarray, mask: np.ndarray, np_rng: np.random.Generator,
             white: bool = True) -> np.ndarray:
    # Zero mask leaves pixels untouched. Rerun controls with this v2 overlay.
    alpha = (.8 * np.clip(mask, 0, 1))[..., None]
    if white:
        texture = np.full_like(image, 245, dtype=np.float32)
    else:
        texture = np_rng.uniform(30, 225, image.shape).astype(np.float32)
    return np.clip(image * (1 - alpha) + texture * alpha, 0, 255).astype(np.uint8)


def random_mask(item, rng):
    # Same size/blending/mode distribution as CorrMask; change position only.
    h, w = item["d1"].shape
    y, x = rng.randrange(h), rng.randrange(w)
    mh = rng.randint(max(16, h//12), max(24, h//4))
    mw = rng.randint(max(24, w//12), max(32, w//4))
    m = np.zeros((h, w), np.float32)
    m[max(0,y-mh//2):min(h,y+mh//2), max(0,x-mw//2):min(w,x+mw//2)] = 1
    return cv2.GaussianBlur(m, (0, 0), 1.2)

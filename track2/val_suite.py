"""Fixed Track 2 local validation suite (backbone-agnostic).

Clean: all 49 held-out frames. Corrupted: the SAME frames, one fixed seed per
(condition, frame), severity 0.5, each view corrupted independently (as in
RobustSpring and Track 3 frozen_eval.corrupt). Corruption taxonomy = Track 3 Q4Aug
families + the Track 3 selection "brightness" (x0.65 exposure) condition.

Metrics (per frame, then mean over frames, as the Spring devkit averages per sample):
  Abs      : Spring-style, prediction on the 1080p grid vs the 4 corresponding 4K GT
             values, min error (stereoflow/criterion.StereoDatasetMetrics special case).
             Valid = all four GT values finite.
  Abs_str  : Track 3 convention, strided GT [::2, ::2], valid = finite (>=0).
  D1       : err>3 and err>5% |gt| using the min-error GT value.   1px: err>1.
  Delta    : mean over ALL pixels of |d_corr - d_clean| (GT-free, RobustSpring-style).
LOCAL_PROXY_RBS = 0.5*CleanAbs/B_S + 0.5*(CleanAbs + DeltaAbs_local)/B_R_proxy.
This is a synthetic local proxy, NOT the official benchmark result.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from . import q4_primitives as q4
from .paths import B_R_PROXY, B_S

SEVERITY = 0.5
BASE_SEED = 20260916
CORRUPTIONS = ("brightness", "color", "blur", "noise", "jpeg", "pixelate", "rain", "snow", "spatter")
GROUPS = {"brightness": "color_exposure", "color": "color_exposure", "blur": "blur", "noise": "noise",
          "jpeg": "compression", "pixelate": "compression", "rain": "weather", "snow": "weather",
          "spatter": "weather"}
# fixed photometric parameters for 'color' (brightness, contrast, saturation, gamma, gray)
COLOR_PARAMS = (1.15, 0.8, 1.3, 0.85, False)


def condition_seed(ci: int, fi: int) -> int:
    return BASE_SEED + 1000 * (ci + 1) + fi


def corrupt_pair(left: np.ndarray, right: np.ndarray, name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    nr = np.random.default_rng(seed)
    rng = random.Random(seed)
    out = []
    for img in (left, right):
        if name == "brightness":
            img = np.clip(img.astype(np.float32) * .65, 0, 255).astype(np.uint8)
        elif name == "color":
            img = q4._photo(img, COLOR_PARAMS)
        elif name in ("rain", "snow", "spatter"):
            m = q4.weather_mask(img.shape[:2], name, SEVERITY, rng, nr)
            img = q4._overlay(img, m, nr, white=True)
        else:
            img = q4._degrade(img, name, SEVERITY, nr)
        out.append(img)
    return out[0], out[1]


def manifest(frames) -> dict:
    return dict(severity=SEVERITY, base_seed=BASE_SEED, corruptions=list(CORRUPTIONS), groups=GROUPS,
                color_params=list(COLOR_PARAMS), per_view_independent=True,
                seeds={c: {f.key: condition_seed(ci, fi) for fi, f in enumerate(frames)}
                       for ci, c in enumerate(CORRUPTIONS)})


@torch.no_grad()
def frame_metrics(pred: torch.Tensor, disp_full: torch.Tensor, disp_str: torch.Tensor) -> dict:
    """pred HxW, disp_full 2Hx2W, disp_str HxW (all float tensors on the same device)."""
    g = torch.stack([disp_full[0::2, 0::2], disp_full[1::2, 0::2], disp_full[0::2, 1::2], disp_full[1::2, 1::2]])
    valid = torch.isfinite(g).all(0)
    err4 = (g - pred[None]).abs()
    err, idx = torch.nan_to_num(err4, nan=1e9).min(0)
    gsel = torch.gather(torch.nan_to_num(g), 0, idx[None])[0]
    e = err[valid]
    n = int(valid.sum())
    vs = torch.isfinite(disp_str) & (disp_str >= 0)
    es = (pred - disp_str)[vs].abs()
    return dict(abs=float(e.mean()), d1=float(((e > 3) & (e > .05 * gsel[valid].abs())).float().mean() * 100),
                px1=float((e > 1).float().mean() * 100), abs_str=float(es.mean()), n=n)


def local_proxy_rbs(clean_abs: float, delta_abs: float) -> float:
    return .5 * clean_abs / B_S + .5 * (clean_abs + delta_abs) / B_R_PROXY


def summarize(rows: list[dict], corruptions=CORRUPTIONS, groups=None) -> dict:
    """rows: dicts with condition, frame, abs, d1, px1, abs_str, delta (None for clean)."""
    out = {}
    groups = groups or GROUPS
    conds = ["clean"] + [c for c in corruptions if any(r["condition"] == c for r in rows)]
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        m = {k: float(np.mean([r[k] for r in rs])) for k in ("abs", "d1", "px1", "abs_str")}
        if c != "clean":
            m["delta"] = float(np.mean([r["delta"] for r in rs]))
            m["delta_valid"] = float(np.mean([r["delta_valid"] for r in rs]))
        m["frames"] = len(rs)
        out[c] = m
    corr = [c for c in conds if c != "clean"]
    clean = out["clean"]
    s = dict(Clean_Abs=clean["abs"], Clean_D1=clean["d1"], Clean_1px=clean["px1"], Clean_Abs_strided=clean["abs_str"],
             per_condition=out, complete=len(corr) == len(corruptions))
    if corr:
        s["Corrupt_Abs"] = float(np.mean([out[c]["abs"] for c in corr]))
        s["Corrupt_D1"] = float(np.mean([out[c]["d1"] for c in corr]))
        s["DeltaAbs_local"] = float(np.mean([out[c]["delta"] for c in corr]))
        s["DeltaAbs_local_validpx"] = float(np.mean([out[c]["delta_valid"] for c in corr]))
        s["LOCAL_PROXY_RBS"] = local_proxy_rbs(s["Clean_Abs"], s["DeltaAbs_local"])
        # diagnostic linear form: RbS = a*CleanAbs + b*DeltaAbs
        s["RBS_linear_coeffs"] = dict(clean=.5 / B_S + .5 / B_R_PROXY, delta=.5 / B_R_PROXY)
        gmap = {}
        for c in corr:
            gmap.setdefault(groups[c], []).append(out[c])
        s["per_group"] = {g: {k: float(np.mean([m[k] for m in ms])) for k in ("abs", "d1", "delta")}
                          for g, ms in gmap.items()}
    return s

"""S2Aug: the stereo-only specialisation of Track 3 Q4Aug.

Track 3 Q4Aug (Roco/scripts/visibility_consistency_core.augment) acts on the quadruplet
(l0, r0, l1, r1):
  * geometry (resize/crop) happens BEFORE Q4Aug and is shared; Q4Aug itself is purely
    photometric/occluder overlay and never modifies labels;
  * gate p=0.7; shared photometric params for all views; one family out of
    blur/noise/jpeg/pixelate/rain/snow/spatter with severity U(.15,.75);
  * weather masks drawn on l0 and TRANSPORTED:   r0 <- splat(m, -d1)              [stereo]
                                                 l1 <- splat(m, flow)             [temporal]
                                                 r1 <- splat(m, flow - d2)        [temporal+stereo]
  * CorrMask (p=.25 inside the gate): box at the GT priority location; world-locked (p=.55,
    transported like weather) or sensor-locked on one of r0/r1/l1.

S2Aug keeps the stereo parts only: views (L, R), transport R <- W_stereo(M_L, d_gt).
Removed: flow/d2 transport, l1/r1 views, temporal masks and all DPFlow dependencies.
Sensor-locked masks can only hit R (the non-reference view), the stereo analogue of Track 3's
r0 choice.

Transport. Track 3 used nearest forward splat with max aggregation + 3x3 dilation and no
visibility reasoning (mode 'track3_max', kept for parity). The default 'zbuffer' mode
excludes invalid/negative disparities and out-of-bounds targets, resolves many-to-one
collisions in favour of the closest surface (largest disparity; any valid source pixel,
masked or not, can occlude), and fills only 1-pixel splat holes (pixels that received no
source) from their 3x3 neighbourhood.

The CorrMask overlay (importance-dependent) is applied on the GPU in stereo_corrmask.py after
the teacher forward; this module decides WHETHER a mask event happens and its world/sensor
mode, so the random stream is fully determined by the sample seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F

from . import q4_primitives as q4

FAMILIES = ("blur", "noise", "jpeg", "pixelate", "rain", "snow", "spatter")
WEATHER = ("rain", "snow", "spatter")


@dataclass
class S2AugConfig:
    p_aug: float = 0.7                  # Track 3 Q4 gate
    families: tuple = FAMILIES
    severity: tuple = (0.15, 0.75)      # Track 3 range
    photometric: bool = True            # Track 3 shared photometric jitter inside the gate
    mode: str = "stereo_shared"         # stereo_shared | stereo_asymmetric | image_plane
    transport: str = "zbuffer"          # zbuffer | track3_max
    occlusion_tol: float = 1.0          # px; sources within tol of the z-buffer winner stay visible
    p_mask: float = 0.25                # Track 3 CorrMask event probability inside the gate
    p_world: float = 0.55               # Track 3 world-locked share
    mask_strategy: str = "none"         # none | random | corrmask (placement decided on GPU)
    # share of weather events drawn as sensor overlays on BOTH views instead of the world-locked transport:
    # identical mask at identical pixel coordinates (zero-parallax distractor, p_overlay_same) or independent masks.
    p_overlay: float = 0.0
    p_overlay_same: float = 0.5
    # region texture erasure (not used by the final model). A world-consistent rectangular region loses ALL texture in both views (filled
    # with its own mean colour + light noise, feathered edge); GT unchanged, so the twin losses must reproduce the
    # clean disparity from the mean colour, the region position and the surrounding context (evidence-free regions,
    # the regime fog/frost create; not tied to any corruption model). Applied inside the gate, after the family.
    p_erase: float = 0.0
    erase_size: tuple = (0.15, 0.6)     # region height/H and width/W range
    erase_noise: float = 2.0            # grey-level noise sigma added to the flat fill (uint8 scale)
    # curriculum (fraction of total steps); values ramp linearly from *_start to target
    warmup_frac: float = 0.0
    p_aug_start: float = 0.35
    severity_hi_start: float = 0.45

    def to_dict(self):
        d = asdict(self)
        d["families"] = list(self.families)
        d["severity"] = list(self.severity)
        return d


def curriculum(cfg: S2AugConfig, progress: float) -> tuple[float, float]:
    if cfg.warmup_frac <= 0 or progress >= cfg.warmup_frac:
        return cfg.p_aug, cfg.severity[1]
    t = progress / cfg.warmup_frac
    return (cfg.p_aug_start + t * (cfg.p_aug - cfg.p_aug_start),
            cfg.severity_hi_start + t * (cfg.severity[1] - cfg.severity_hi_start))


# ------------------------------------------------------------------ transport
def transport_left_to_right(mask: torch.Tensor, disp: torch.Tensor, mode: str = "zbuffer",
                            occlusion_tol: float = 1.0, fill_holes: bool = True) -> torch.Tensor:
    """Warp a left-view occluder mask into the right view of a rectified pair.

    mask, disp: (H,W) or (B,1,H,W); disp = positive left disparity, x_R = x_L - d.
    Returns a mask of the same shape. Never modifies inputs.
    """
    squeeze = mask.ndim == 2
    m = mask[None, None] if squeeze else mask
    d = disp[None, None] if disp.ndim == 2 else disp
    B, _, H, W = m.shape
    m = m.float()
    d = d.float()
    xx = torch.arange(W, device=m.device).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(H, device=m.device).view(1, 1, H, 1).expand(B, 1, H, W)
    bb = torch.arange(B, device=m.device).view(B, 1, 1, 1).expand(B, 1, H, W)
    if mode == "track3_max":
        # exact torch port of q4.forward_splat_mask(m, -d, 0)
        tx = torch.round(xx.float() + torch.nan_to_num(-d)).long()  # rint == round-half-even
        valid = (tx >= 0) & (tx < W) & (m > 0)
        flat = bb * H * W + yy * W + tx
        out = torch.zeros(B * H * W, device=m.device)
        out.scatter_reduce_(0, flat[valid], m[valid], reduce="amax")
        out = out.view(B, 1, H, W)
        out = F.max_pool2d(F.pad(out, (1, 1, 1, 1), value=0.0), 3, 1)  # cv2.dilate 3x3 (zero border)
    elif mode == "zbuffer":
        src = torch.isfinite(d) & (d >= 0)
        tx = torch.round(xx.float() - torch.where(src, d, torch.zeros_like(d))).long()
        src = src & (tx >= 0) & (tx < W)
        flat = bb * H * W + yy * W + tx
        zbuf = torch.full((B * H * W,), -1.0, device=m.device)
        zbuf.scatter_reduce_(0, flat[src], d[src], reduce="amax")
        visible = src & (d >= zbuf[flat.clamp(0, B * H * W - 1)].view_as(d) - occlusion_tol)
        sel = visible & (m > 0)
        out = torch.zeros(B * H * W, device=m.device)
        out.scatter_reduce_(0, flat[sel], m[sel], reduce="amax")
        hit = (zbuf >= 0).view(B, 1, H, W)
        out = out.view(B, 1, H, W)
        if fill_holes:
            neigh = F.max_pool2d(F.pad(out, (1, 1, 1, 1), value=0.0), 3, 1)
            out = torch.where(hit, out, neigh)
    elif mode == "image_plane":
        out = m.clone()
    else:
        raise ValueError(mode)
    return out[0, 0] if squeeze else out


def transport_np(mask: np.ndarray, disp: np.ndarray, cfg: S2AugConfig) -> np.ndarray:
    if cfg.mode == "image_plane":
        return mask.copy()
    return transport_left_to_right(torch.from_numpy(mask), torch.from_numpy(disp), cfg.transport,
                                   cfg.occlusion_tol).numpy()


# ------------------------------------------------------------------- S2Aug
def s2aug(left: np.ndarray, right: np.ndarray, disp: np.ndarray, rng: random.Random,
          nr: np.random.Generator, cfg: S2AugConfig, progress: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Photometric/occluder corruption of a rectified pair. `disp` is read-only.

    Returns (left~, right~, info). info['mask_event'] / info['mask_world'] tell the GPU stage
    whether to place a CorrMask/random box and whether it is world- or sensor-locked.
    """
    L, R = left.copy(), right.copy()
    p_aug, sev_hi = curriculum(cfg, progress)
    info = dict(aug="clean", kind="none", severity=0.0, mask_event=0, mask_world=0,
                weather_area_L=0.0, weather_area_R=0.0, overlay="none", erase=0, erase_area=0.0)
    if rng.random() < p_aug:
        if cfg.photometric:
            pL = q4._photo_params(rng)
            pR = q4._photo_params(rng) if cfg.mode == "stereo_asymmetric" else pL
            L, R = q4._photo(L, pL), q4._photo(R, pR)
        kind = rng.choice(tuple(cfg.families))
        severity = rng.uniform(cfg.severity[0], max(cfg.severity[0], sev_hi))
        sev_R = rng.uniform(cfg.severity[0], max(cfg.severity[0], sev_hi)) if cfg.mode == "stereo_asymmetric" else severity
        info.update(aug="s2:" + kind, kind=kind, severity=severity)
        if kind.startswith("r20:"):
            # RobustSpring-name corruption operators (track2/robust20.py); S2Aug severity 0.15..0.75 -> ImageNet-C level 1..5.
            # Independent per-view realisation; depth-aware fog uses the left GT disparity for both views (approximation).
            from . import robust20
            to_level = lambda u: int(min(5, max(1, 1 + int(u / 0.75 * 4.999))))
            name = kind[4:]
            L = robust20.apply(name, L, rng.randrange(2**31), disp=disp, severity=to_level(severity))
            R = robust20.apply(name, R, rng.randrange(2**31), disp=disp, severity=to_level(sev_R))
        elif kind in WEATHER:
            mL = q4.weather_mask(disp.shape, kind, severity, rng, nr)
            if cfg.mode == "stereo_asymmetric":
                mR = q4.weather_mask(disp.shape, kind, sev_R, rng, nr)  # independent sensor weather
            elif cfg.p_overlay > 0 and rng.random() < cfg.p_overlay:    # extra draws only when enabled (stream-stable)
                if rng.random() < cfg.p_overlay_same:
                    mR = mL.copy()                                       # zero-parallax overlay (lens frost / dirt)
                    info["overlay"] = "same"
                else:
                    mR = q4.weather_mask(disp.shape, kind, severity, rng, nr)
                    info["overlay"] = "indep"
            else:
                mR = transport_np(mL, disp, cfg)
            L, R = q4._overlay(L, mL, nr, white=True), q4._overlay(R, mR, nr, white=True)
            info.update(weather_area_L=float((mL > .05).mean()), weather_area_R=float((mR > .05).mean()))
        else:
            L, R = q4._degrade(L, kind, severity, nr), q4._degrade(R, kind, sev_R, nr)
        if cfg.mask_strategy != "none" and rng.random() < cfg.p_mask:
            info["mask_event"] = 1
            info["mask_world"] = int(rng.random() < cfg.p_world)
        if cfg.p_erase > 0 and rng.random() < cfg.p_erase:          # extra draws only when enabled (stream-stable)
            L, R, area = erase_region(L, R, disp, rng, nr, cfg)
            info.update(erase=1, erase_area=area)
    info["mask_seed"] = rng.randrange(2**31)
    return L, R, info


def erase_region(left: np.ndarray, right: np.ndarray, disp: np.ndarray, rng: random.Random, nr: np.random.Generator,
                 cfg: S2AugConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Texture erasure of one world-locked rectangle: left box -> transported to the right view with the GT disparity;
    each view's region is replaced by its own mean colour (+ noise), feathered edge. Returns (L~, R~, left area)."""
    import cv2
    H, W = disp.shape
    h = max(8, int(H * rng.uniform(*cfg.erase_size)))
    w = max(8, int(W * rng.uniform(*cfg.erase_size)))
    y, x = rng.randrange(0, H - h + 1), rng.randrange(0, W - w + 1)
    mL = np.zeros((H, W), np.float32)
    mL[y:y + h, x:x + w] = 1.0
    mL = cv2.GaussianBlur(mL, (0, 0), 3)
    mR = transport_np(mL, disp, cfg)

    def fill(img, m):
        sel = m > 0.5
        if sel.sum() < 64:
            return img
        flat = np.empty(img.shape, np.float32)
        flat[:] = img[sel].reshape(-1, 3).mean(0)
        flat += nr.normal(0.0, cfg.erase_noise, flat.shape).astype(np.float32)
        out = img.astype(np.float32) * (1.0 - m[..., None]) + flat * m[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    return fill(left, mL), fill(right, mR), float((mL > 0.5).mean())


# ------------------------------------------------------------ geometry (Track 3)
def resize_crop(left, right, disp, crop, rng: random.Random, min_scale=.9, max_scale=1.15):
    """Track 3 corresguard_core.resize_and_crop restricted to (L, R, d): shared log-uniform
    scale, bilinear images, nearest disparity scaled by sx, shared random crop."""
    import cv2
    h, w = disp.shape
    scale = math.exp(rng.uniform(math.log(min_scale), math.log(max_scale)))
    nh, nw = max(crop[0], round(h * scale)), max(crop[1], round(w * scale))
    sx = nw / w
    L = cv2.resize(left, (nw, nh), interpolation=cv2.INTER_LINEAR)
    R = cv2.resize(right, (nw, nh), interpolation=cv2.INTER_LINEAR)
    D = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_NEAREST) * sx
    y, x = rng.randrange(nh - crop[0] + 1), rng.randrange(nw - crop[1] + 1)
    s = (slice(y, y + crop[0]), slice(x, x + crop[1]))
    return (np.ascontiguousarray(L[s]), np.ascontiguousarray(R[s]), np.ascontiguousarray(D[s]))

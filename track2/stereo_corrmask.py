"""Stereo-CorrMask: importance-weighted placement of occluder boxes for stereo.

Track 3 CorrMask (corresguard_core._priority_mask) is MODEL-AGNOSTIC and GT-based:
    score = .45*|grad d1| + .30*|flow| + .25*|grad I_l0|   (each /p95, clipped)
and places ONE box at argmax(score + U(0,.05)).  Box size h/12..h/4 x w/12..w/4, blurred
(sigma 1.2), random texture overlay with alpha .8, world- (transported) or sensor-locked.

Track 2 keeps the box geometry / blending / world-sensor logic and replaces the placement:
    P(center) = alpha * P_importance + (1 - alpha) * P_uniform,
    P_importance ∝ (region-smoothed importance)^(1/temperature)
with pluggable ImportanceProviders (all detached, all training-only):
    random              uniform (control)
    gt_priority         Track 3 signal without the flow term: (.45 edge(d) + .25 detail(I_L)) / .70
    teacher_confidence  exp(-|d_T - d_gt| / tau) * detail(I_L)  (reliable AND informative regions)
    croco_attention     concentration 1 - H/log(Nk) of the teacher's left->right decoder
                        cross-attention, per 16x16 left token, upsampled to pixels
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from .s2aug import transport_left_to_right


@dataclass
class CorrMaskConfig:
    provider: str = "random"            # random | gt_priority | teacher_confidence | croco_attention
    alpha: float = 0.6                  # importance share of the centre distribution
    temperature: float = 0.5            # <1 sharpens importance
    tau: float = 1.0                    # px, teacher-confidence scale
    n_regions: int = 1                  # Track 3: one box
    min_hw_frac: tuple = (1 / 12, 1 / 12)
    max_hw_frac: tuple = (1 / 4, 1 / 4)
    blur_sigma: float = 1.2
    alpha_overlay: float = 0.8
    smooth_frac: float = 1 / 8          # region scale for importance smoothing (of min(H,W))

    def to_dict(self):
        d = asdict(self)
        d["min_hw_frac"] = list(self.min_hw_frac)
        d["max_hw_frac"] = list(self.max_hw_frac)
        return d


def _norm95(x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    B = x.shape[0]
    flat = x.reshape(B, -1)
    if valid is not None:
        flat = torch.where(valid.reshape(B, -1), flat, torch.zeros_like(flat))
    q = torch.quantile(flat.float()[:, ::7], .95, dim=1).clamp_min(1e-6).view(B, 1, 1, 1)
    return (x / q).clamp(0, 1)


def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kx.transpose(2, 3))
    return torch.sqrt(gx * gx + gy * gy)


def detail_map(left255: torch.Tensor) -> torch.Tensor:
    gray = (0.299 * left255[:, :1] + 0.587 * left255[:, 1:2] + 0.114 * left255[:, 2:3])
    return _norm95(_sobel_mag(gray))


@torch.no_grad()
def importance(provider: str, left255: torch.Tensor, gt: torch.Tensor, teacher_disp: torch.Tensor | None = None,
               attn_tokens: torch.Tensor | None = None, tau: float = 1.0) -> torch.Tensor:
    """All inputs B x C x H x W on one device. Returns B x 1 x H x W in [0, 1]."""
    valid = torch.isfinite(gt) & (gt >= 0)
    if provider == "random":
        return torch.ones_like(gt)
    if provider == "gt_priority":
        g = torch.nan_to_num(gt, nan=0.0, posinf=0.0)
        imp = (.45 * _norm95(_sobel_mag(g)) + .25 * detail_map(left255)) / .70
    elif provider == "teacher_confidence":
        assert teacher_disp is not None
        err = (teacher_disp.detach() - torch.nan_to_num(gt, nan=0.0, posinf=0.0)).abs()
        imp = torch.exp(-err / tau) * detail_map(left255)
    elif provider == "croco_attention":
        assert attn_tokens is not None  # B x Nq (16x16 patches)
        B, _, H, W = gt.shape
        tok = attn_tokens.detach().float().view(B, 1, H // 16, W // 16)
        tok = (tok - tok.amin((2, 3), keepdim=True)) / (tok.amax((2, 3), keepdim=True) - tok.amin((2, 3), keepdim=True)).clamp_min(1e-6)
        imp = F.interpolate(tok, size=(H, W), mode="bilinear", align_corners=False)
    else:
        raise ValueError(provider)
    return (imp * valid).clamp(0, 1)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(round(sigma * 4 * 2 + 1)) | 1  # cv2 kernel size rule for float images
    r = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-r * r / (2 * sigma * sigma))
    g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


@torch.no_grad()
def sample_masks(imp: torch.Tensor, seeds: list[int], cfg: CorrMaskConfig) -> tuple[torch.Tensor, list[dict]]:
    """Stochastic importance-weighted box placement. imp: B1HW. Returns (B1HW mask in [0,1], infos)."""
    B, _, H, W = imp.shape
    masks = torch.zeros_like(imp)
    k = max(3, int(min(H, W) * cfg.smooth_frac) | 1)
    smooth = F.avg_pool2d(imp, k, 1, k // 2, count_include_pad=False)
    infos = []
    for b in range(B):
        gen = torch.Generator(device="cpu").manual_seed(int(seeds[b]))
        p_imp = smooth[b, 0].flatten().double().clamp_min(0) ** (1.0 / cfg.temperature)
        p = torch.full_like(p_imp, 1.0 / p_imp.numel())
        if cfg.provider != "random" and p_imp.sum() > 0:
            p = cfg.alpha * p_imp / p_imp.sum() + (1 - cfg.alpha) * p
        centers = torch.multinomial(p.cpu(), cfg.n_regions, replacement=True, generator=gen)
        m = torch.zeros(H, W, device=imp.device, dtype=imp.dtype)
        for c in centers.tolist():
            y, x = divmod(c, W)
            lo_h, lo_w = max(16, int(H * cfg.min_hw_frac[0])), max(24, int(W * cfg.min_hw_frac[1]))
            hi_h, hi_w = max(24, int(H * cfg.max_hw_frac[0])), max(32, int(W * cfg.max_hw_frac[1]))
            mh = int(torch.randint(lo_h, hi_h + 1, (1,), generator=gen))
            mw = int(torch.randint(lo_w, hi_w + 1, (1,), generator=gen))
            m[max(0, y - mh // 2):min(H, y + mh // 2), max(0, x - mw // 2):min(W, x + mw // 2)] = 1
        masks[b, 0] = m
        infos.append(dict(centers=centers.tolist()))
    if cfg.blur_sigma > 0:
        masks = _gaussian_blur(masks, cfg.blur_sigma).clamp(0, 1)
    return masks, infos


@torch.no_grad()
def apply_mask_events(left255, right255, gt, masks, world, event, seeds, cfg: CorrMaskConfig,
                      transport="zbuffer", occlusion_tol=1.0):
    """Random-texture overlay (Track 3 _overlay, white=False). world: B bool (transport to R);
    sensor-locked events corrupt only R with the untransported box. Returns new tensors + stats."""
    B, _, H, W = left255.shape
    L, R = left255.clone(), right255.clone()
    mR_all = torch.zeros_like(masks)
    for b in range(B):
        if not bool(event[b]):
            continue
        gen = torch.Generator(device="cpu").manual_seed(int(seeds[b]) + 7919)
        m = masks[b:b + 1]
        a = (cfg.alpha_overlay * m)
        if bool(world[b]):
            texL = torch.empty(1, 3, H, W).uniform_(30, 225, generator=gen).to(L)
            L[b:b + 1] = L[b:b + 1] * (1 - a) + texL * a
            mR = transport_left_to_right(m, gt[b:b + 1], mode=transport, occlusion_tol=occlusion_tol)
        else:
            mR = m
        texR = torch.empty(1, 3, H, W).uniform_(30, 225, generator=gen).to(R)
        aR = cfg.alpha_overlay * mR
        R[b:b + 1] = R[b:b + 1] * (1 - aR) + texR * aR
        mR_all[b:b + 1] = mR
    # uint8 quantisation as in Track 3 (np.clip(...).astype(uint8))
    return L.clamp(0, 255).floor(), R.clamp(0, 255).floor(), mR_all


@torch.no_grad()
def mask_stats(imp: torch.Tensor, masks: torch.Tensor, event: torch.Tensor) -> dict:
    sel = event.bool()
    if not sel.any():
        return dict(mask_area=0.0, imp_in=float("nan"), imp_out=float("nan"))
    m = masks[sel] > .5
    i = imp[sel]
    return dict(mask_area=float(m.float().mean()), imp_in=float(i[m].mean()) if m.any() else float("nan"),
                imp_out=float(i[~m].mean()))

"""Track 2 losses. rho = Track 3 robust penalty (visibility_consistency_core.robust_error):
rho(e) = sqrt(e^2 + 0.01) - 0.1.

  L_delta  = sum_p w_delta rho(d_corr - sg(d_clean)) / sum_p w_delta,
             w_delta = valid                       (delta_weight='valid')
                     = valid * exp(-|d_T - gt|/tau_delta)   (delta_weight='teacher')
  L_anchor = sum_p w_anchor rho(d_clean - sg(d_T)) / sum_p w_anchor,
             w_anchor = valid * clamp(exp(-|d_T - gt|/tau_anchor), w_min, 1)   (detached)
  L_feat   = sum_l mean_p [ w_feat (1 - cos(z_corr, sg(z_clean))) ]  over left decoder tokens
All weights are detached. Weighted means return 0 (with a live graph) when sum w == 0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rho(e: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(e * e + .01) - .1


def valid_gt(gt: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(gt) & (gt >= 0)


def weighted_mean(value: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w = w.detach().to(value.dtype)
    s = w.sum()
    if s <= 0:
        return (value * 0).sum()
    return (value * w).sum() / s


@torch.no_grad()
def teacher_confidence(d_teacher: torch.Tensor, gt: torch.Tensor, tau: float) -> torch.Tensor:
    v = valid_gt(gt)
    err = (d_teacher.detach().float() - torch.nan_to_num(gt.float(), nan=0.0, posinf=0.0)).abs()
    return torch.where(v, torch.exp(-err / tau), torch.zeros_like(err))


def delta_loss(d_corr, d_clean, gt, weight="valid", d_teacher=None, tau=1.0):
    v = valid_gt(gt).float()
    w = v if weight == "valid" else v * teacher_confidence(d_teacher, gt, tau)
    return weighted_mean(rho(d_corr - d_clean.detach()), w), w


def anchor_loss(d_clean, d_teacher, gt, tau=1.0, w_min=0.0):
    v = valid_gt(gt).float()
    c = teacher_confidence(d_teacher, gt, tau)
    w = v * c.clamp(min=w_min, max=1.0)
    return weighted_mean(rho(d_clean - d_teacher.detach()), w), w


def feature_loss(z_corr: list[torch.Tensor], z_clean: list[torch.Tensor], w_pix: torch.Tensor, patch: int = 16):
    """z_*: lists of B x N x C left-view decoder tokens (same token order: raster over the
    left crop). w_pix: B1HW pixel weight, pooled to the token grid."""
    B, _, H, W = w_pix.shape
    wt = F.avg_pool2d(w_pix.float(), patch).view(B, -1)  # B x N
    total = 0.0
    for zc, zk in zip(z_corr, z_clean):
        assert zc.shape == zk.shape and zc.shape[1] == wt.shape[1], (zc.shape, wt.shape)
        cos = F.cosine_similarity(F.normalize(zc.float(), dim=-1), F.normalize(zk.detach().float(), dim=-1), dim=-1)
        total = total + weighted_mean(1 - cos, wt)
    return total / max(1, len(z_corr))


def croco_native_loss(criterion, pred, conf, gt):
    """CroCo LaplacianLossBounded2 exactly as in stereoflow/engine.train_one_epoch. Non-finite GT
    becomes +inf, as upstream _read_hdf5_disp does, so masked positions have finite gradients."""
    g = torch.where(torch.isfinite(gt), gt, torch.full_like(gt, float("inf")))
    return criterion(pred.float(), g.float(), conf.float())

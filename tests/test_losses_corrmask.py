import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__import__("os").environ.get("ROCO_TRACK2_WORK", ROOT))   # checkpoints/, outputs/, third_party/croco live here
sys.path.insert(0, str(ROOT))
from track2 import losses as Ls  # noqa: E402
from track2.stereo_corrmask import CorrMaskConfig, importance, sample_masks, apply_mask_events  # noqa: E402


def gt_field(B=2, H=32, W=64):
    g = torch.rand(B, 1, H, W) * 20
    g[:, :, 0, :4] = float("nan")
    return g


def test_delta_zero_positive_invalid():
    gt = gt_field()
    d = torch.rand_like(gt) * 10
    l0, _ = Ls.delta_loss(d.clone().requires_grad_(), d, gt)
    assert l0.abs() < 1e-6
    l1, _ = Ls.delta_loss(d + 1.0, d, gt)
    assert l1 > 0.5
    pert = d.clone()
    pert[:, :, 0, :4] += 100  # invalid pixels only
    l2, _ = Ls.delta_loss(pert, d, gt)
    assert l2.abs() < 1e-6


def test_delta_no_grad_through_clean_target():
    gt = gt_field()
    dc = (torch.rand_like(gt) * 10).requires_grad_()
    dx = (torch.rand_like(gt) * 10).requires_grad_()
    l, _ = Ls.delta_loss(dx, dc, gt, weight="teacher", d_teacher=torch.rand_like(gt), tau=1.0)
    l.backward()
    assert dc.grad is None and dx.grad is not None and dx.grad.abs().sum() > 0


def test_anchor_zero_and_weights_and_no_teacher_grad():
    gt = gt_field()
    dt = gt.nan_to_num(0).clone().requires_grad_()
    l0, w = Ls.anchor_loss(dt.detach().clone(), dt, gt, tau=1.0)
    assert l0.abs() < 1e-6
    close = gt.nan_to_num(0) + 0.1
    far = gt.nan_to_num(0) + 10
    wc = Ls.teacher_confidence(close, gt, 1.0)
    wf = Ls.teacher_confidence(far, gt, 1.0)
    v = Ls.valid_gt(gt)
    assert wc[v].min() > 0.9 and wf[v].max() < 1e-3 and wc[~v].sum() == 0
    dc = torch.rand_like(gt).requires_grad_()
    l, _ = Ls.anchor_loss(dc, dt, gt, tau=1.0)
    l.backward()
    assert dt.grad is None and dc.grad.abs().sum() > 0


def test_croco_native_loss_finite_grads_with_nan_gt():
    sys.path.insert(0, str(WORK / "third_party/croco"))
    from stereoflow.criterion import LaplacianLossBounded2
    gt = gt_field()
    p = torch.rand_like(gt).requires_grad_()
    c = torch.rand_like(gt).requires_grad_()
    l = Ls.croco_native_loss(LaplacianLossBounded2(), p, c, gt)
    l.backward()
    assert torch.isfinite(l) and torch.isfinite(p.grad).all() and torch.isfinite(c.grad).all()


def test_feature_loss_zero_and_positive():
    z = [torch.randn(2, 8, 16)]
    w = torch.ones(2, 1, 32, 64)
    assert Ls.feature_loss(z, [z[0].clone()], w) < 1e-6
    assert Ls.feature_loss([torch.randn(2, 8, 16)], z, w) > 0.1


def _imp_fixture(H=128, W=256):
    imp = torch.zeros(1, 1, H, W)
    imp[:, :, 20:60, 150:230] = 1.0
    return imp


def test_corrmask_area_and_determinism():
    cfg = CorrMaskConfig(provider="gt_priority", alpha=1.0)
    imp = _imp_fixture()
    a, _ = sample_masks(imp, [5], cfg)
    b, _ = sample_masks(imp, [5], cfg)
    assert torch.equal(a, b)
    areas = [float((sample_masks(imp, [s], cfg)[0] > .5).float().mean()) for s in range(50)]
    mean_area = sum(areas) / len(areas)
    # Track 3 box: E[h]~H/6, E[w]~W/6 -> ~1/36 of the image (less at borders)
    assert 0.01 < mean_area < 0.05, mean_area


def test_corrmask_prefers_importance_over_random():
    imp = _imp_fixture()
    def mean_inside(cfg):
        vals = []
        for s in range(100):
            m, _ = sample_masks(imp, [s], cfg)
            sel = m > .5
            vals.append(float(imp[sel].mean()))
        return sum(vals) / len(vals)
    cm = mean_inside(CorrMaskConfig(provider="teacher_confidence", alpha=0.6))
    rnd = mean_inside(CorrMaskConfig(provider="random"))
    assert cm > rnd + 0.2, (cm, rnd)


def test_importance_providers_shapes_detached():
    B, H, W = 1, 64, 128
    left = torch.rand(B, 3, H, W) * 255
    gt = gt_field(B, H, W)
    for prov, kw in [("random", {}), ("gt_priority", {}), ("teacher_confidence", dict(teacher_disp=torch.rand(B, 1, H, W))),
                     ("croco_attention", dict(attn_tokens=torch.rand(B, (H // 16) * (W // 16))))]:
        i = importance(prov, left, gt, **kw)
        assert i.shape == (B, 1, H, W) and not i.requires_grad and torch.isfinite(i).all()
        assert i.min() >= 0 and i.max() <= 1


def test_apply_mask_world_vs_sensor_and_gt_untouched():
    B, H, W = 2, 64, 128
    L = torch.full((B, 3, H, W), 100.0)
    R = torch.full((B, 3, H, W), 100.0)
    gt = torch.full((B, 1, H, W), 8.0)
    gt0 = gt.clone()
    masks = torch.zeros(B, 1, H, W)
    masks[:, :, 20:40, 40:80] = 1
    cfg = CorrMaskConfig()
    L2, R2, mR = apply_mask_events(L, R, gt, masks, world=torch.tensor([1, 0]), event=torch.tensor([1, 1]),
                                   seeds=[1, 2], cfg=cfg)
    assert (L2[0] != 100).any() and torch.equal(L2[1], L[1])     # sensor-locked: left untouched
    assert torch.equal(mR[0, 0, 20:40, 32:72], torch.ones(20, 40))  # world: shifted by d=8
    assert torch.equal(mR[1], masks[1])
    assert torch.equal(gt, gt0)

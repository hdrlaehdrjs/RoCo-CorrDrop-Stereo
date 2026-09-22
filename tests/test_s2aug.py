import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from track2.s2aug import S2AugConfig, s2aug, transport_left_to_right, resize_crop  # noqa: E402


def box(H=40, W=80, y=(10, 20), x=(30, 45)):
    m = torch.zeros(H, W)
    m[y[0]:y[1], x[0]:x[1]] = 1
    return m


def test_A_zero_disparity_aligned():
    m = box()
    out = transport_left_to_right(m, torch.zeros_like(m))
    assert torch.equal(out, m)


def test_B_constant_disparity_shift():
    m = box()
    out = transport_left_to_right(m, torch.full_like(m, 5.0))
    exp = torch.zeros_like(m)
    exp[10:20, 25:40] = 1
    assert torch.equal(out, exp)


def test_B2_fractional_disparity_rounds():
    m = box()
    out = transport_left_to_right(m, torch.full_like(m, 4.6))
    exp = torch.zeros_like(m)
    exp[10:20, 25:40] = 1
    assert torch.equal(out, exp)


def test_C_invalid_disparity_excluded():
    m = box()
    d = torch.full_like(m, 5.0)
    d[10:20, 30:35] = float("nan")
    d[10:20, 35:37] = -3.0
    d[10:20, 37:38] = float("inf")
    out = transport_left_to_right(m, d, fill_holes=False)
    assert out[10:20, 25:30].sum() == 0 and out[10:20, 30:33].sum() == 0  # sources 30..37 dropped
    assert torch.equal(out[10:20, 33:40], torch.ones(10, 7))
    assert torch.isfinite(out).all()


def test_D_out_of_bounds_removed():
    m = box(x=(0, 20))
    out = transport_left_to_right(m, torch.full_like(m, 15.0))
    assert torch.equal(out[10:20, 0:5], torch.ones(10, 5))
    assert out.sum() == 50  # 15 columns fell outside, nothing wrapped
    assert out[:, -20:].sum() == 0


def test_occlusion_closer_surface_wins():
    H, W = 20, 60
    m = torch.zeros(H, W)
    m[:, 20:30] = 1                     # far surface (d=2) masked
    d = torch.full((H, W), 2.0)
    d[:, 30:40] = 12.0                  # near surface (d=12), unmasked, lands on 18..27
    out = transport_left_to_right(m, d)
    assert out[:, 18:28].sum() == 0     # occluded by the unmasked closer surface
    # Track 3 max-splat would paint the mask onto the occluder
    t3 = transport_left_to_right(m, d, mode="track3_max")
    assert t3[:, 18:28].sum() > 0


def test_zbuffer_fills_one_pixel_holes_only():
    H, W = 10, 60
    m = torch.zeros(H, W)
    m[:, 10:30] = 1
    d = torch.zeros(H, W)
    d[:, 20:] = 1.0                      # stretch creates one unhit column at x=19
    out = transport_left_to_right(m, d)
    assert out[:, 19].sum() == H


def _pair(h=64, w=128, seed=0):
    r = np.random.default_rng(seed)
    L = r.integers(0, 255, (h, w, 3), dtype=np.uint8)
    R = r.integers(0, 255, (h, w, 3), dtype=np.uint8)
    d = r.uniform(0, 20, (h, w)).astype(np.float32)
    d[0, :5] = np.nan
    return L, R, d


@pytest.mark.parametrize("mode", ["stereo_shared", "stereo_asymmetric", "image_plane"])
def test_E_deterministic_and_F_shapes_and_G_gt_unchanged(mode):
    L, R, d = _pair()
    d0 = d.copy()
    cfg = S2AugConfig(p_aug=1.0, mode=mode, mask_strategy="random", p_mask=1.0)
    for fam in ("blur", "noise", "jpeg", "pixelate", "rain", "snow", "spatter"):
        cfg.families = (fam,)
        a = s2aug(L, R, d, random.Random(3), np.random.default_rng(3), cfg)
        b = s2aug(L, R, d, random.Random(3), np.random.default_rng(3), cfg)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
        assert a[0].shape == L.shape and a[1].shape == R.shape and a[0].dtype == np.uint8
        assert np.array_equal(d, d0, equal_nan=True)
        assert not np.array_equal(a[0], L) or fam in ("rain", "snow", "spatter")


def test_G_resize_crop_scales_gt_consistently():
    L, R, d = _pair(200, 400)
    l, r, dd = resize_crop(L, R, np.nan_to_num(d), (64, 128), random.Random(1))
    assert l.shape == (64, 128, 3) and dd.shape == (64, 128)


def test_H_no_temporal_or_flow_dependency():
    # modules present at interpreter start (e.g. site-packages editable finders) are not imports
    code = ("import sys; base=set(sys.modules); sys.path.insert(0, %r); import track2.s2aug, track2.stereo_corrmask, track2.losses;"
            "bad=[m for m in set(sys.modules)-base if any(k in m.lower() for k in ('ptlflow','dpflow','flow_io','corresguard','visibility_consistency','defom','track3'))];"
            "print(bad); assert not bad" % str(ROOT))
    subprocess.run([sys.executable, "-c", code], check=True)
    src = (ROOT / "track2/s2aug.py").read_text()
    body = src.split('"""', 2)[2]  # skip module docstring which documents what was removed
    for word in ("flow", "d2", "l1", "r1", "temporal"):
        assert word not in body.replace("float", "").replace("overflow", ""), word


def test_curriculum_ramp():
    from track2.s2aug import curriculum
    cfg = S2AugConfig(warmup_frac=.1)
    assert curriculum(cfg, 0.0) == (cfg.p_aug_start, cfg.severity_hi_start)
    assert curriculum(cfg, .5) == (cfg.p_aug, cfg.severity[1])


def _changed(a, b):
    return np.abs(a.astype(int) - b.astype(int)).sum(-1) > 0


def _iou(m1, m2):
    return (m1 & m2).sum() / max(1, (m1 | m2).sum())


def test_g3_overlay_modes_zero_parallax_independent_and_default_transport():
    L, R, d = _pair(96, 192, seed=2)
    d = np.full_like(d, 12.0)
    same = S2AugConfig(p_aug=1.0, families=("rain",), photometric=False, p_overlay=1.0, p_overlay_same=1.0)
    for s in range(5):
        Lx, Rx, info = s2aug(L, R, d, random.Random(s), np.random.default_rng(s), same)
        mL, mR = _changed(Lx, L), _changed(Rx, R)
        assert info["overlay"] == "same" and mL.any() and _iou(mL, mR) > 0.9
    indep = S2AugConfig(p_aug=1.0, families=("rain",), photometric=False, p_overlay=1.0, p_overlay_same=0.0)
    Lx, Rx, info = s2aug(L, R, d, random.Random(1), np.random.default_rng(1), indep)
    assert info["overlay"] == "indep" and _iou(_changed(Lx, L), _changed(Rx, R)) < 0.5
    # default: world-locked transport (mask shifted by the disparity), and the random stream of existing runs is unchanged
    off = S2AugConfig(p_aug=1.0, families=("rain",), photometric=False)
    Lx, Rx, info = s2aug(L, R, d, random.Random(1), np.random.default_rng(1), off)
    mL, mR = _changed(Lx, L), _changed(Rx, R)
    shifted = np.zeros_like(mL)
    shifted[:, :-12] = mL[:, 12:]
    assert info["overlay"] == "none" and _iou(shifted, mR) > 0.8 and _iou(mL, mR) < _iou(shifted, mR)
    ref = s2aug(L, R, d, random.Random(1), np.random.default_rng(1), S2AugConfig(p_aug=1.0, families=("rain",), photometric=False, p_overlay=0.0))
    assert np.array_equal(ref[0], Lx) and np.array_equal(ref[1], Rx)


def test_g4_erase_region_flat_world_locked_and_stream_stable():
    L, R, d = _pair(96, 192, seed=5)
    d = np.full_like(d, 12.0)
    cfg = S2AugConfig(p_aug=1.0, families=("blur",), photometric=False, severity=(0.15, 0.16), p_erase=1.0, erase_noise=0.0)
    ref = S2AugConfig(p_aug=1.0, families=("blur",), photometric=False, severity=(0.15, 0.16))
    for s in range(5):
        Lx, Rx, info = s2aug(L, R, d, random.Random(s), np.random.default_rng(s), cfg)
        L0, R0, _ = s2aug(L, R, d, random.Random(s), np.random.default_rng(s), ref)
        assert info["erase"] == 1 and 0.02 < info["erase_area"] < 0.4
        mL, mR = _changed(Lx, L0), _changed(Rx, R0)
        # flat interior: one colour covers most of the erased box (feathered rim excluded via the 0.5 threshold area)
        import cv2
        deep = cv2.erode(mL.astype(np.uint8), np.ones((51, 51), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if deep.any():
            px = Lx[deep].reshape(-1, 3).astype(int)
            assert (np.abs(px - px[0]).max(1) <= 1).all()                     # exactly flat up to rounding
        shifted = np.zeros_like(mL)
        shifted[:, :-12] = mL[:, 12:]
        assert _iou(shifted, mR) > 0.7                                      # world-locked: shifted by the disparity
        assert np.array_equal(d, np.full_like(d, 12.0))
    a = s2aug(L, R, d, random.Random(3), np.random.default_rng(3), ref)
    b = s2aug(L, R, d, random.Random(3), np.random.default_rng(3), S2AugConfig(p_aug=1.0, families=("blur",), photometric=False, severity=(0.15, 0.16), p_erase=0.0))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2]["mask_seed"] == b[2]["mask_seed"]
    # real crop size: the deep interior of the erased box is one colour in both views
    Lb, Rb, db = _pair(352, 704, seed=9)
    db = np.full_like(db, 20.0)
    checked = 0
    for s in range(6):
        Lx, Rx, info = s2aug(Lb, Rb, db, random.Random(s), np.random.default_rng(s), cfg)
        L0, R0, _ = s2aug(Lb, Rb, db, random.Random(s), np.random.default_rng(s), ref)
        for img, base in ((Lx, L0), (Rx, R0)):
            deep = cv2.erode(_changed(img, base).astype(np.uint8), np.ones((51, 51), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)   # rim at the image border is not "deep"
            if deep.any():
                px = img[deep].reshape(-1, 3).astype(int)
                assert (np.abs(px - px[0]).max(1) <= 1).all()
                checked += 1
    assert checked >= 4


def test_r20_families_geometry_and_determinism():
    L, R, d = _pair()
    d0 = d.copy()
    for fam in ("r20:gaussian_noise", "r20:fog", "r20:motion_blur", "r20:contrast", "r20:jpeg_compression"):
        cfg = S2AugConfig(p_aug=1.0, families=(fam,))
        a = s2aug(L, R, d, random.Random(4), np.random.default_rng(4), cfg)
        b = s2aug(L, R, d, random.Random(4), np.random.default_rng(4), cfg)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        assert a[0].shape == L.shape and a[0].dtype == np.uint8 and np.array_equal(d, d0, equal_nan=True)

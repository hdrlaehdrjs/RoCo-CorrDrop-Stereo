import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
T3 = Path(__import__("os").environ.get("ROCO_TRACK3_ROOT", Path(__file__).resolve().parents[1] / "third_party/track3"))
sys.path.insert(0, str(ROOT))
from track2.config import load_config  # noqa: E402


def test_final_configs_differ_from_full_only_by_the_ablated_setting():
    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            o.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
        return o
    full = flat(load_config("track2_croco_x7_x6b_mono"))
    expect = {"track2_croco_abl_no_dropout": {"corrdrop.p"}, "track2_croco_abl_no_delta": {"loss.lambda_delta"},
              "track2_croco_abl_randkey_s1": {"corrdrop.keys"}, "track2_croco_abl_image_plane": {"s2aug.transport"},
              "track2_croco_abl_no_teacher": {"loss.lambda_anchor", "loss.delta_weight", "corrmask.provider"},
              "track2_croco_abl_seed2": {"seed"}, "track2_croco_abl_randkey_s2": {"corrdrop.keys", "seed"}}
    for name, keys in expect.items():
        c = flat(load_config(name))
        diff = {k for k in set(full) | set(c) if full.get(k) != c.get(k)} - {"name", "config_file"}
        assert diff == keys, (name, diff)
    assert load_config("track2_croco_x7_x6b_mono", ["loss.lambda_anchor=0.3"])["loss"]["lambda_anchor"] == 0.3


def test_defom_native_loss_matches_track3_supervision():
    sys.path[:0] = [str(T3 / "scripts")]
    try:
        import visibility_consistency_core as vc
        from track2.backbones import DEFOMBackbone, Out
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Track 3 / DEFOM import unavailable: {e}")
    bb = DEFOMBackbone({})
    gt = torch.rand(2, 1, 16, 32) * 30
    gt[:, :, :2, :3] = float("nan")
    gt[:, :, 5, 5] = 0.0  # valid sky
    preds = [torch.rand_like(gt) * 30 for _ in range(8)]
    ours = bb.native_loss(Out(disp=preds[-1], payload=preds), gt)
    ref = vc.supervision(preds, gt)
    assert torch.allclose(ours, ref, rtol=1e-6, atol=1e-6), (ours, ref)

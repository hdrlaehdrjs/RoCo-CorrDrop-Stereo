"""Monocular fallback head unit tests on the tiny CPU CroCo-Stereo of tests/test_care.py."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__import__("os").environ.get("ROCO_TRACK2_WORK", ROOT))   # checkpoints/, outputs/, third_party/croco live here
sys.path[:0] = [str(ROOT), str(WORK / "third_party/croco")]
from tests.test_care import tiny, pair, TINY_ARGS, H, W  # noqa: E402
from track2.mono import MonoConfig, install_mono, mono_disabled, mono_parameters  # noqa: E402
from track2.care import CAREConfig, install_care, care_keydrop, care_parameters  # noqa: E402
from track2.croco_model import build_croco_stereo  # noqa: E402
from track2.backbones import Out  # noqa: E402

HOOKS = [0, 0, 1, 1]   # tiny encoder has 2 blocks; the real model uses [5, 11, 17, 23]


def test_mono_install_near_identity_left_only_and_disable():
    net = tiny()
    l, r = pair()
    with torch.no_grad():
        ref = net(l, r)
    keys0 = set(net.state_dict())
    mono = install_mono(net, MonoConfig(hooks=HOOKS, fuse_bias_init=20.0))
    keys1 = set(net.state_dict())
    assert keys0 < keys1 and all(k.startswith("mono.") for k in keys1 - keys0)
    assert mono.n_copied > 0 and len(mono_parameters(net)) == len(list(mono.parameters()))
    with torch.no_grad():
        out = net(l, r)
    assert (out - ref).abs().max() < 1e-4 and float(mono.last["w"].min()) > 0.999
    assert mono.last["d_mono"].shape == ref[:, :1].shape and torch.isfinite(mono.last["d_mono"]).all()
    with torch.no_grad(), mono_disabled(net):
        assert (net(l, r) - ref).abs().max() == 0
    # the mono prediction depends on the left image only; the stereo prediction on both
    r2 = torch.randn_like(r)
    with torch.no_grad():
        net(l, r2)
        d_mono2, d_st2 = mono.last["d_mono"].clone(), mono.last["d_st"].clone()
        net(l, r)
    assert torch.equal(mono.last["d_mono"], d_mono2) and (mono.last["d_st"] - d_st2).abs().max() > 1e-4
    # default init: w = sigmoid(6)
    net3 = tiny()
    m3 = install_mono(net3, MonoConfig(hooks=HOOKS))
    with torch.no_grad():
        net3(l, r)
    assert abs(float(m3.last["w"].mean()) - 0.9975) < 1e-3


def test_mono_fusion_gradients_and_checkpoint_roundtrip(tmp_path):
    net = tiny()
    mono = install_mono(net, MonoConfig(hooks=HOOKS, fuse_bias_init=0.0))   # w = 0.5: both branches active
    l, r = pair()
    out = net(l, r)
    assert abs(float(mono.last["w"].mean()) - 0.5) < 1e-6
    fused = 0.5 * mono.last["d_st"] + 0.5 * mono.last["d_mono"]
    assert (out[:, :1] - fused).abs().max() < 1e-5
    (out.mean() + mono.last["d_mono"].abs().mean()).backward()
    grads = {n: p.grad for n, p in mono.named_parameters()}
    # DPT leaves refinenet4.resConfUnit1 unused (single input at the top level): same set as in the stereo head
    st_none = {n for n, p in net.head.named_parameters() if p.grad is None}
    assert all(n.replace("head.", "", 1) in st_none for n, g in grads.items() if g is None)
    assert all(torch.isfinite(g).all() for g in grads.values() if g is not None)
    assert float(grads["fuse.2.weight"].abs().sum()) > 0
    assert all(float(g.abs().sum()) > 0 for n, g in grads.items() if g is not None and n.startswith("head.") and g.ndim > 1)
    # detach_tokens (default): the mono loss alone never reaches the encoder; the stereo path still does
    net.zero_grad(set_to_none=True)
    net(l, r)
    mono.last["d_mono"].abs().mean().backward()
    assert all(p.grad is None for p in net.enc_blocks.parameters())
    net.zero_grad(set_to_none=True)
    net(l, r)[:, :1].mean().backward()
    assert any(p.grad is not None for p in net.enc_blocks.parameters())
    with torch.no_grad():
        for p in mono.parameters():
            p.add_(torch.randn_like(p) * 0.01)
        a = net(l, r)
    ck = tmp_path / "ck.pt"
    torch.save(dict(model=net.state_dict(), args=dict(croco_args=TINY_ARGS, crop=[H, W], criterion="LaplacianLossBounded2()",
                                                      tile_conf_mode="conf_expsigmoid_15_3", task="stereo"), mono=mono.cfg.to_dict()), ck)
    m, meta = build_croco_stereo(str(ck), "cpu")
    assert meta["mono"]["hooks"] == HOOKS and sorted(m.net.state_dict()) == sorted(net.state_dict())
    with torch.no_grad():
        b = m.net(l, r)
    assert (a - b).abs().max() < 1e-6
    assert torch.equal(m.net.state_dict()["mono.fuse.2.bias"], net.state_dict()["mono.fuse.2.bias"])


def test_mono_coexists_with_care_fallback_and_correspondence_dropout():
    net = tiny()
    install_care(net, CAREConfig(variant="fallback", block=0, blocks=[2, 5]))
    mono = install_mono(net, MonoConfig(hooks=HOOKS))
    l, r = pair()
    keys = set(net.state_dict())
    assert any(k.startswith("care.") for k in keys) and any(k.startswith("mono.") for k in keys)
    assert len(care_parameters(net)) == 10 and len(mono_parameters(net)) > 10
    qdrop = torch.zeros(2, 32, dtype=torch.bool)
    qdrop[:, :8] = True
    with torch.no_grad(), care_keydrop(net, qdrop):
        out = net(l, r)
    assert torch.isfinite(out).all() and mono.last["w"].shape == (2, 1, H, W)
    o = Out(disp=out[:, :1], payload=out[:, 1:2], extra=dict(mono.last))
    assert set(o.extra) == {"d_st", "c_st", "d_mono", "c_mono", "w"}

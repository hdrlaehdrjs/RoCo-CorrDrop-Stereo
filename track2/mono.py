"""Monocular fallback head for CroCo-Stereo: a second DPT head on the LEFT encoder tokens only, fused per pixel
with the stereo (decoder) prediction by a learned reliability weight

    d = w * d_stereo + (1 - w) * d_mono,      w = sigmoid(Conv(f) + b),
    f (detached) = [d_stereo/64, d_mono/64, conf_stereo/3, conf_mono/3, log1p |d_stereo - d_mono|].

Zero-initialised output conv and b = fuse_bias_init (sigmoid(6) = 0.9975): near-identity at init. The mono head gets
its own Laplacian loss (lambda_mono in the trainer) so it becomes competent while w is still ~1; every other loss acts
on the fused output. Parameters live in net.mono (state_dict prefix 'mono.'), so plain checkpoints still load
strictly; checkpoints saved with a mono head carry ckpt['mono'] and are rebuilt by build_croco_stereo.
Inference: same single forward; +1 DPT head (~30 M parameters on top of 447 M), no test-time switching.
The mono head only ever sees left-view tokens (encoder hooks < enc_depth), so it cannot use correspondence.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn


@dataclass
class MonoConfig:
    hooks: list = field(default_factory=lambda: [5, 11, 17, 23])   # left ENCODER blocks feeding the mono DPT head
    fuse_hidden: int = 16
    fuse_bias_init: float = 6.0          # sigmoid(6) = 0.9975: stereo-dominant at init
    init_from_stereo_head: bool = True   # copy shape-compatible DPT weights (refinenets, output convs) from the stereo head
    detach_tokens: bool = True           # mono head reads the encoder tokens without back-propagating into the encoder
                                         # (its Laplacian loss is O(100) at init and must not disturb the stereo encoder)

    def to_dict(self):
        return asdict(self)


class MonoFusion(nn.Module):
    def __init__(self, cfg: MonoConfig, croconet):
        super().__init__()
        from models.head_downstream import PixelwiseTaskWithDPT   # third_party/croco (on sys.path via track2.croco_model)
        assert max(cfg.hooks) < croconet.enc_depth and len(cfg.hooks) == 4, "mono hooks must be 4 left-encoder blocks"
        self.cfg = cfg
        self.head = PixelwiseTaskWithDPT(hooks_idx=list(cfg.hooks), num_channels=2)
        self.head.setup(croconet)
        self.fuse = nn.Sequential(nn.Conv2d(5, cfg.fuse_hidden, 3, padding=1), nn.GELU(), nn.Conv2d(cfg.fuse_hidden, 1, 3, padding=1))
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.constant_(self.fuse[-1].bias, cfg.fuse_bias_init)
        self.n_copied = 0
        if cfg.init_from_stereo_head:
            src, dst = croconet.head.state_dict(), self.head.state_dict()
            copied = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
            self.head.load_state_dict({**dst, **copied})
            self.n_copied = len(copied)
        self.enabled = True
        self.last: dict = {}


def mono_fused_forward(stereo_head, mono: MonoFusion, tokens, img_info):
    st = type(stereo_head).forward(stereo_head, tokens, img_info)        # original PixelwiseTaskWithDPT.forward
    if not mono.enabled:
        return st
    mo = mono.head([t.detach() for t in tokens] if mono.cfg.detach_tokens else tokens, img_info)
    d_st, c_st, d_mo, c_mo = st[:, :1], st[:, 1:2], mo[:, :1], mo[:, 1:2]
    with torch.no_grad():
        f = torch.cat([d_st / 64, d_mo / 64, c_st / 3, c_mo / 3, (d_st - d_mo).abs().log1p()], 1)
    w = torch.sigmoid(mono.fuse(f.to(st.dtype)))
    mono.last = dict(d_st=d_st, c_st=c_st, d_mono=d_mo, c_mono=c_mo, w=w)
    return torch.cat([w * d_st + (1 - w) * d_mo, w * c_st + (1 - w) * c_mo], 1)


def install_mono(net: nn.Module, cfg: MonoConfig) -> MonoFusion:
    """Attach the mono head + fuser as net.mono and route net.head through the fusion (original head untouched)."""
    assert getattr(net, "mono", None) is None, "mono head already installed"
    mono = MonoFusion(cfg, net).to(next(net.head.parameters()).device)
    net.mono = mono
    head = net.head
    head.forward = lambda tokens, img_info: mono_fused_forward(head, mono, tokens, img_info)
    return mono


def mono_parameters(net):
    mono = getattr(net, "mono", None)
    return [] if mono is None else list(mono.parameters())


@contextmanager
def mono_disabled(net):
    mono = getattr(net, "mono", None)
    if mono is None:
        yield
        return
    old = mono.enabled
    mono.enabled = False
    try:
        yield
    finally:
        mono.enabled = old

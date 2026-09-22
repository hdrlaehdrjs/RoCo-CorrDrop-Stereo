"""Backbone adapters for Track 2 training. Exactly one backbone is imported per process
(CroCo and DEFOM both expose top-level `utils`-style modules).

Adapter contract (all images are float RGB 0..255, B3HW; gt B1HW with NaN for invalid):
  build(checkpoint)                 -> student model on cuda
  build_teacher(checkpoint)         -> frozen model (eval, requires_grad False)
  configure_trainable(model, scope) -> scope in {decoder_head, last_k:<k>, full}
  param_groups(model, lr, lr_encoder, wd)
  train_mode(model)
  can_share_encoder(student, teacher)
  encode(model, L, R)               -> opaque, no_grad (only when the encoder is frozen)
  forward(model, L, R, enc=None, want_feats=None, want_attn=False) -> Out
  native_loss(out, gt)
  infer(model, L, R)                -> B1HW, the single-model inference path
  save(model, path, meta)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .paths import TRACK3_ROOT
from . import losses as Ls


@dataclass
class Out:
    disp: torch.Tensor
    payload: object = None
    feats: list = field(default_factory=list)
    attn: torch.Tensor | None = None
    extra: dict = field(default_factory=dict)   # CroCo mono fusion: d_st, c_st, d_mono, c_mono, w (with graph)


def _frozen(model: nn.Module) -> nn.Module:
    model.eval()
    model.requires_grad_(False)
    return model


class CroCoBackbone:
    name = "croco"
    feat_patch = 16

    def __init__(self, cfg: dict):
        from .croco_model import build_croco_stereo, normalize_rgb255
        from stereoflow.criterion import LaplacianLossBounded2
        self._build = build_croco_stereo
        self.norm = normalize_rgb255
        self.criterion = LaplacianLossBounded2()  # checkpoint criterion: LaplacianLossBounded2(a=3,b=3)
        self.overlap = cfg.get("inference", {}).get("overlap", 0.7)
        self.attn_blocks = tuple(cfg.get("corrmask", {}).get("attn_blocks", [-1]))
        self.attn_sup_blocks = tuple(cfg.get("attn_sup", {}).get("blocks", [])) if cfg.get("attn_sup") else ()
        cc = dict(cfg.get("care") or {})
        self.care_lr_scale = float(cc.pop("lr_scale", 1.0))
        self.care_cfg = cc or None   # dropout | fallback on the student's decoder cross-attention (trained jointly)
        mc = dict(cfg.get("mono") or {})
        self.mono_lr_scale = float(mc.pop("lr_scale", 1.0))
        mc.pop("lambda_mono", None)
        self.mono_cfg = mc or None   # monocular fallback head fused with the stereo prediction

    def build(self, ck, student=True):
        m, meta = self._build(str(ck), "cuda")
        assert "a=3" in meta["criterion"].replace(" ", "").replace(".", "") or "()" in meta["criterion"]
        if not student:
            return m
        from .care import CAREConfig, install_care
        if self.care_cfg:
            if getattr(m.net, "care", None) is None:
                install_care(m.net, CAREConfig(**self.care_cfg))
                m.net.care.to(next(m.net.dec_blocks.parameters()).device)   # installed after the model moved to cuda
            else:
                assert m.net.care.cfg.to_dict() == CAREConfig(**self.care_cfg).to_dict(), "checkpoint CARE config differs"
        elif self.attn_sup_blocks:
            # parameter-free probe on the original cross-attention (direction 1: attention-level supervision)
            install_care(m.net, CAREConfig(variant="probe", block=self.attn_sup_blocks[0],
                                           blocks=list(self.attn_sup_blocks) if len(self.attn_sup_blocks) > 1 else None))
        if self.mono_cfg:
            from .mono import MonoConfig, install_mono
            if getattr(m.net, "mono", None) is None:
                install_mono(m.net, MonoConfig(**self.mono_cfg))
            else:
                assert m.net.mono.cfg.to_dict() == MonoConfig(**self.mono_cfg).to_dict(), "checkpoint mono config differs"
        return m

    def build_teacher(self, ck):
        return _frozen(self.build(ck, student=False))

    def configure_trainable(self, m, scope: str):
        m.requires_grad_(True)
        if scope == "decoder_head":
            m.freeze_encoder()
        elif scope.startswith("last_k:"):
            m.unfreeze_last_k_encoder_blocks(int(scope.split(":")[1]))
        elif scope != "full":
            raise ValueError(scope)
        return m.trainable_summary()

    def param_groups(self, m, lr, lr_encoder, wd):
        groups = m.get_trainable_parameter_groups(lr, lr_encoder, wd)
        from .care import care_parameters
        from .mono import mono_parameters

        def split_out(params, scope, lr_s, wd_s):
            # move `params` out of the timm-style groups into their own group(s) with their own learning rate
            ids = {id(p) for p in params}
            nonlocal groups
            for g in groups:
                keep = [(p, n) for p, n in zip(g["params"], g["names"]) if id(p) not in ids]
                g["params"], g["names"] = [p for p, _ in keep], [n for _, n in keep]
            groups = [g for g in groups if g["params"]]
            for decay in (True, False):
                sel = [(n, p) for n, p in m.net.named_parameters() if id(p) in ids and (not (p.ndim == 1 or n.endswith(".bias"))) == decay]
                if sel:
                    groups.append(dict(params=[p for _, p in sel], names=[n for n, _ in sel], weight_decay=wd_s if decay else 0.0,
                                       lr=lr_s, base_lr=lr_s, scope=scope))

        care = [p for p in care_parameters(m.net) if p.requires_grad]
        if care:   # gate / no-correspondence embedding: tiny modules, no weight decay, scaled learning rate
            split_out(care, "care", lr * self.care_lr_scale, 0.0)
        mono = [p for p in mono_parameters(m.net) if p.requires_grad]
        if mono:   # mono DPT head + fuser, trained from (partially copied) init: scaled learning rate
            split_out(mono, "mono", lr * self.mono_lr_scale, wd)
        return groups

    def train_mode(self, m):
        m.train()

    def can_share_encoder(self, student, teacher):
        if teacher is None or not student.encoder_fully_frozen():
            return False
        for a, b in zip([p for mm in student.encoder_modules() for p in mm.parameters()],
                        [p for mm in teacher.encoder_modules() for p in mm.parameters()]):
            if not torch.equal(a, b):
                return False
        return True

    @torch.no_grad()
    def encode(self, m, L, R):
        return m.encode(self.norm(L), self.norm(R))

    def forward(self, m, L, R, enc=None, want_feats=None, want_attn=False):
        l, r = self.norm(L), self.norm(R)
        if enc is None:
            enc = m.encode(l, r)
        if want_attn:
            with m.capture_cross_attention_concentration(self.attn_blocks) as store:
                disp, conf, dec = m.decode(enc, l.shape, return_dec_feats=True)
            attn = torch.stack(store).mean(0)
        else:
            disp, conf, dec = m.decode(enc, l.shape, return_dec_feats=True)
            attn = None
        feats = [dec[i] for i in want_feats] if want_feats else []
        mono = getattr(m.net, "mono", None)
        extra = dict(mono.last) if mono is not None and mono.enabled else {}
        return Out(disp=disp, payload=conf, feats=feats, attn=attn, extra=extra)

    def native_loss(self, out: Out, gt):
        return Ls.croco_native_loss(self.criterion, out.disp, out.payload, gt)

    def mono_loss(self, out: Out, gt):
        """Laplacian loss of the mono head alone (keeps it competent while the fusion weight is ~1)."""
        return Ls.croco_native_loss(self.criterion, out.extra["d_mono"], out.extra["c_mono"], gt)

    @torch.no_grad()
    def infer(self, m, L, R):
        return m.tiled_predict(L, R, overlap=self.overlap, tile_batch=1)

    def save(self, m, path, meta):
        from .care import care_parameters
        from .mono import mono_parameters
        extra = {}
        if care_parameters(m.net):
            extra["care"] = m.net.care.cfg.to_dict()   # rebuilt automatically by build_croco_stereo (parameter variants only)
        if mono_parameters(m.net):
            extra["mono"] = m.net.mono.cfg.to_dict()
        torch.save(dict(model=m.net.state_dict(), args=dict(croco_args=m.meta["croco_args"], crop=m.meta["crop"],
                        criterion=m.meta["criterion"], tile_conf_mode=m.meta["tile_conf_mode"], task="stereo"),
                        track2=meta, **extra), path)


class DEFOMBackbone:
    """DEFOM-Stereo exactly as trained in Track 3 (paper_ablation_train stereo branch)."""
    name = "defom"

    def __init__(self, cfg: dict):
        root = TRACK3_ROOT / "third_party/DEFOM-Stereo"
        sys.path[:0] = [str(root), str(root / "core")]
        from core.defom_stereo import DEFOMStereo
        from core.utils.utils import InputPadder
        self.DEFOMStereo, self.InputPadder = DEFOMStereo, InputPadder
        t = cfg.get("defom", {})
        self.train_iters, self.train_scale_iters = t.get("train_iters", 8), t.get("train_scale_iters", 4)
        self.test_iters, self.test_scale_iters = t.get("test_iters", 16), t.get("test_scale_iters", 6)

    @staticmethod
    def _args():
        from argparse import Namespace
        return Namespace(dinov2_encoder="vits", idepth_scale=0.5, hidden_dims=[128] * 3, corr_implementation="reg",
                         shared_backbone=False, corr_levels=2, corr_radius=4,
                         scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], scale_corr_radius=2,
                         n_downsample=2, context_norm="batch", n_gru_layers=3, mixed_precision=False)

    def build(self, ck):
        m = self.DEFOMStereo(self._args())
        state = torch.load(ck, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "stereo" in state and "model" not in state:
            state = state["stereo"]
        m.load_state_dict(state.get("model", state) if isinstance(state, dict) else state, strict=True)
        return m.cuda()

    def build_teacher(self, ck):
        return _frozen(self.build(ck))

    def configure_trainable(self, m, scope):
        m.requires_grad_(True)
        enc = m.defomencoder
        if scope == "decoder_head":
            enc.requires_grad_(False)
        elif scope.startswith("last_k:"):
            enc.requires_grad_(False)
            k = int(scope.split(":")[1])
            pre = enc.depth_anything.pretrained
            for b in list(pre.blocks)[-k:]:
                b.requires_grad_(True)
            pre.norm.requires_grad_(True)
        elif scope != "full":
            raise ValueError(scope)
        tot = sum(p.numel() for p in m.parameters())
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        et = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        return dict(total=tot, trainable=tr, encoder_trainable=et, decoder_head_trainable=tr - et,
                    encoder_total=sum(p.numel() for p in enc.parameters()), encoder_frozen=et == 0)

    def param_groups(self, m, lr, lr_encoder, wd):
        enc_ids = {id(p) for p in m.defomencoder.parameters()}
        heads = [p for p in m.parameters() if p.requires_grad and id(p) not in enc_ids]
        enc = [p for p in m.parameters() if p.requires_grad and id(p) in enc_ids]
        g = [dict(params=heads, lr=lr, base_lr=lr, weight_decay=wd, scope="decoder_head")]
        if enc:
            g.append(dict(params=enc, lr=lr_encoder, base_lr=lr_encoder, weight_decay=wd, scope="encoder"))
        return g

    def train_mode(self, m):
        m.train()
        m.freeze_bn()
        if not any(p.requires_grad for p in m.defomencoder.parameters()):
            m.defomencoder.eval()

    def can_share_encoder(self, student, teacher):
        return False

    def encode(self, m, L, R):
        return None

    feat_patch = 4  # fnet matching features are at 1/4 resolution

    def forward(self, m, L, R, enc=None, want_feats=None, want_attn=False):
        """want_feats (any non-empty value): left matching features fmap1 = fnet(...)[0], the
        same tensor Track 3's S4_feature variant regularised, as B x N x C tokens."""
        assert not want_attn, "attention importance is CroCo-only"
        store = []
        h = m.fnet.register_forward_hook(lambda mod, i, o: store.append(o[0])) if want_feats else None
        try:
            preds = m(L, R, iters=self.train_iters, scale_iters=self.train_scale_iters, test_mode=False)
        finally:
            if h is not None:
                h.remove()
        feats = [store[0].flatten(2).transpose(1, 2)] if want_feats else []
        return Out(disp=preds[-1], payload=preds, feats=feats)

    def native_loss(self, out: Out, gt):
        # Track 3 visibility_consistency_core.supervision (vendored; see tests/test_backbones.py)
        preds = out.payload
        n = len(preds)
        weights = [.9 ** (15 * (n - 1 - i) / max(1, n - 1)) for i in range(n)]
        valid = Ls.valid_gt(gt)
        tot = 0
        for w, p in zip(weights, preds):
            e = Ls.rho(p - torch.nan_to_num(gt))
            tot = tot + w * (e[valid].mean() if valid.any() else e.sum() * 0)
        return tot / sum(weights)

    @torch.no_grad()
    def infer(self, m, L, R):
        pad = self.InputPadder(L.shape, divis_by=32)
        l, r = pad.pad(L, R)
        return pad.unpad(m(l, r, iters=self.test_iters, scale_iters=self.test_scale_iters, test_mode=True)).float()

    def save(self, m, path, meta):
        torch.save(dict(model=m.state_dict(), track2=meta), path)


def make_backbone(cfg: dict):
    return {"croco": CroCoBackbone, "defom": DEFOMBackbone}[cfg["backbone"]](cfg)

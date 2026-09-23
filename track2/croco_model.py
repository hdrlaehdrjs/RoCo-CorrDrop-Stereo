"""Thin wrapper around the official CroCo-Stereo implementation (third_party/croco).

Conventions established by inspecting upstream code (commit pinned in
third_party/croco-source.json):
  * input: RGB uint8 -> /255 -> ImageNet mean/std (stereoflow/datasets_stereo.img_to_tensor)
  * img1 = left, img2 = right; output channel 0 = positive left disparity in input
    pixels at input resolution, channel 1 = raw confidence logit for
    LaplacianLossBounded2 (stereoflow/criterion.py)
  * the network is built for a fixed crop (352, 704); full images are processed by
    overlapping tiles with confidence weighting (stereoflow/engine.tiled_pred)
  * decoder: 12 DecoderBlocks, left tokens (queries) cross-attend to right tokens
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from .paths import add_croco_to_path

add_croco_to_path()
from models.croco_downstream import CroCoDownstreamBinocular  # noqa: E402
from models.head_downstream import PixelwiseTaskWithDPT  # noqa: E402
from stereoflow.criterion import LaplacianLossBounded2  # noqa: E402,F401
from stereoflow.engine import _overlapping, _crop  # noqa: E402

IN1K_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IN1K_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_rgb255(x: torch.Tensor) -> torch.Tensor:
    """BCHW float RGB in [0,255] -> CroCo input normalisation."""
    return (x / 255.0 - IN1K_MEAN.to(x)) / IN1K_STD.to(x)


def build_croco_stereo(checkpoint: str, device="cpu", care_cfg: dict | None = None) -> tuple["Track2CroCoStereo", dict]:
    """care_cfg: install CARE (zero-initialised) on top of the loaded weights. CARE checkpoints saved by the Track 2
    trainer carry their CARE config in ckpt['care'] and are rebuilt automatically (E0 weights + CARE weights)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    croco_args = dict(args.croco_args) if hasattr(args, "croco_args") else dict(args["croco_args"])
    criterion = args.criterion if hasattr(args, "criterion") else args["criterion"]
    assert "LaplacianLossBounded2" in criterion, criterion
    head = PixelwiseTaskWithDPT()
    head.num_channels = 2  # disparity + confidence
    net = CroCoDownstreamBinocular(head, **croco_args)
    saved_care = ckpt.get("care") if isinstance(ckpt, dict) else None
    saved_mono = ckpt.get("mono") if isinstance(ckpt, dict) else None
    if saved_care is not None:
        from .care import CAREConfig, install_care
        install_care(net, CAREConfig(**saved_care))
    if saved_mono is not None:
        from .mono import MonoConfig, install_mono
        install_mono(net, MonoConfig(**saved_mono))          # before the strict load: 'mono.*' keys are in the checkpoint
    net.load_state_dict(ckpt["model"], strict=True)
    if saved_care is None and care_cfg:
        from .care import CAREConfig, install_care
        install_care(net, CAREConfig(**care_cfg))
    meta = dict(croco_args=croco_args, criterion=criterion,
                crop=list(args.crop if hasattr(args, "crop") else args["crop"]),
                tile_conf_mode=args.tile_conf_mode if hasattr(args, "tile_conf_mode") else args["tile_conf_mode"])
    meta["care"] = saved_care or care_cfg
    meta["mono"] = saved_mono
    model = Track2CroCoStereo(net, meta).to(device)
    return model, meta


class Track2CroCoStereo(nn.Module):
    """Single CroCo-Stereo model. Inference path = forward/tiled_predict only."""

    def __init__(self, net: CroCoDownstreamBinocular, meta: dict):
        super().__init__()
        self.net = net
        self.meta = meta
        self.crop = tuple(meta["crop"])
        self._attn_store = None

    # ------------------------------------------------------------------ forward
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """left/right: normalised BCHW crops. Returns (disparity B1HW, conf_logit B1HW)."""
        out = self.net(left, right)
        return out[:, :1], out[:, 1:2]

    def encode(self, left, right):
        return self.net.encode_image_pairs(left, right, return_all_blocks=True)

    def decode(self, enc, left_shape, return_dec_feats=False):
        """Decoder + DPT head from precomputed encoder outputs (identical to net.forward)."""
        out, out2, pos, pos2 = enc
        decout = self.net._decoder(out[-1], pos, None, out2, pos2, return_all_blocks=True)
        feats = decout
        allout = out + decout
        H, W = left_shape[-2:]
        pred = self.net.head(allout, {"height": H, "width": W})
        if return_dec_feats:
            return pred[:, :1], pred[:, 1:2], feats
        return pred[:, :1], pred[:, 1:2]

    def decode_frozen_prefix(self, enc, left_shape, first_trainable_block: int, return_dec_feats=False):
        """Same computation as decode(), but decoder blocks before `first_trainable_block` (all frozen, CARE absent)
        run under no_grad. Used for CARE training, where only modules at/after that block need a graph."""
        out, out2, pos, pos2 = enc
        n = self.net
        with torch.no_grad():
            f1 = n.decoder_embed(out[-1])
            f2 = n.decoder_embed(out2)
            if n.dec_pos_embed is not None:
                f1, f2 = f1 + n.dec_pos_embed, f2 + n.dec_pos_embed
            dec, x, y = [], f1, f2
            for blk in n.dec_blocks[:first_trainable_block]:
                x, y = blk(x, y, pos, pos2)
                dec.append(x)
        x = x.detach()
        for blk in n.dec_blocks[first_trainable_block:]:
            x, y = blk(x, y, pos, pos2)
            dec.append(x)
        dec[-1] = n.dec_norm(dec[-1])
        H, W = left_shape[-2:]
        pred = n.head(out + dec, {"height": H, "width": W})
        if return_dec_feats:
            return pred[:, :1], pred[:, 1:2], dec
        return pred[:, :1], pred[:, 1:2]

    def forward_with_features(self, left, right):
        enc = self.encode(left, right)
        return self.decode(enc, left.shape, return_dec_feats=True)

    # --------------------------------------------------------- parameter scopes
    def encoder_modules(self):
        n = self.net
        return [n.patch_embed, n.enc_blocks, n.enc_norm]

    def freeze_encoder(self):
        for m in self.encoder_modules():
            m.requires_grad_(False)

    def unfreeze_last_k_encoder_blocks(self, k: int):
        self.freeze_encoder()
        if k > 0:
            for blk in self.net.enc_blocks[-k:]:
                blk.requires_grad_(True)
            self.net.enc_norm.requires_grad_(True)

    def encoder_fully_frozen(self) -> bool:
        return not any(p.requires_grad for m in self.encoder_modules() for p in m.parameters())

    def get_trainable_parameter_groups(self, lr_decoder: float, lr_encoder: float, weight_decay: float):
        """timm-style: no weight decay for biases/norms; separate encoder LR."""
        enc_ids = {id(p) for m in self.encoder_modules() for p in m.parameters()}
        groups = {}
        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            scope = "encoder" if id(p) in enc_ids else "decoder_head"
            decay = not (p.ndim == 1 or name.endswith(".bias"))
            key = f"{scope}_{'decay' if decay else 'no_decay'}"
            g = groups.setdefault(key, dict(params=[], names=[], weight_decay=weight_decay if decay else 0.0,
                                            lr=lr_encoder if scope == "encoder" else lr_decoder,
                                            base_lr=lr_encoder if scope == "encoder" else lr_decoder, scope=scope))
            g["params"].append(p)
            g["names"].append(name)
        return list(groups.values())

    def trainable_summary(self) -> dict:
        enc_ids = {id(p) for m in self.encoder_modules() for p in m.parameters()}
        s = dict(total=0, trainable=0, encoder_trainable=0, decoder_head_trainable=0, encoder_total=0)
        for p in self.net.parameters():
            s["total"] += p.numel()
            if id(p) in enc_ids:
                s["encoder_total"] += p.numel()
            if p.requires_grad:
                s["trainable"] += p.numel()
                s["encoder_trainable" if id(p) in enc_ids else "decoder_head_trainable"] += p.numel()
        s["encoder_frozen"] = s["encoder_trainable"] == 0
        return s

    # ------------------------------------------------- cross-attention capture
    @contextmanager
    def capture_cross_attention_concentration(self, blocks=(-1,)):
        """Records, per left token, 1 - H(attn)/log(Nk) of the left->right cross-attention,
        averaged over heads and the chosen decoder blocks. No weights are changed."""
        dec = self.net.dec_blocks
        idx = [b % len(dec) for b in blocks]
        store = []
        originals = {}

        def make_forward(ca):
            def fwd(query, key, value, qpos, kpos):
                B, Nq, C = query.shape
                Nk, Nv = key.shape[1], value.shape[1]
                q = ca.projq(query).reshape(B, Nq, ca.num_heads, C // ca.num_heads).permute(0, 2, 1, 3)
                k = ca.projk(key).reshape(B, Nk, ca.num_heads, C // ca.num_heads).permute(0, 2, 1, 3)
                v = ca.projv(value).reshape(B, Nv, ca.num_heads, C // ca.num_heads).permute(0, 2, 1, 3)
                if ca.rope is not None:
                    q = ca.rope(q, qpos)
                    k = ca.rope(k, kpos)
                attn = ((q @ k.transpose(-2, -1)) * ca.scale).softmax(dim=-1)
                with torch.no_grad():
                    a = attn.float().clamp_min(1e-12)
                    ent = -(a * a.log()).sum(-1).mean(1)  # B x Nq
                    store.append(1.0 - ent / math.log(Nk))
                x = (ca.attn_drop(attn) @ v).transpose(1, 2).reshape(B, Nq, C)
                return ca.proj_drop(ca.proj(x))
            return fwd

        try:
            for i in idx:
                ca = dec[i].cross_attn
                originals[i] = ca.forward
                ca.forward = make_forward(ca)
            yield store
        finally:
            for i, f in originals.items():
                dec[i].cross_attn.forward = f

    # ----------------------------------------------------------- tiled predict
    @torch.no_grad()
    def tiled_predict(self, left255: torch.Tensor, right255: torch.Tensor, overlap: float = 0.7,
                      tile_batch: int = 8, amp_dtype: torch.dtype | None = None,
                      conf_mode: str | None = None) -> torch.Tensor:
        """Batched re-implementation of stereoflow.engine.tiled_pred (numerically equivalent
        up to float kernel differences; see tests/gpu_croco_tiling_parity.py). Input RGB 0..255 B3HW."""
        conf_mode = conf_mode or self.meta["tile_conf_mode"]
        assert conf_mode.startswith("conf_expsigmoid_")
        beta, betasig = map(float, conf_mode[len("conf_expsigmoid_"):].split("_"))
        img1, img2 = normalize_rgb255(left255.float()), normalize_rgb255(right255.float())
        B, _, H, W = img1.shape
        wh, ww = self.crop
        assert H >= wh and W >= ww, "images smaller than crop are not used in Spring"
        accu_pred = img1.new_zeros((B, 1, H, W))
        accu_conf = img1.new_zeros((B, H, W)) + 1e-16
        slots = [(sy, sx) for sy in _overlapping(H, wh, overlap) for sx in _overlapping(W, ww, overlap)]
        for i in range(0, len(slots), tile_batch):
            chunk = slots[i:i + tile_batch]
            a = torch.cat([_crop(img1, sy, sx) for sy, sx in chunk], 0)
            b = torch.cat([_crop(img2, sy, sx) for sy, sx in chunk], 0)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = self.net(a, b)
            out = out.float()
            pred, pconf = out[:, :1], out[:, 1:2]
            conf = torch.exp(-beta * 2 * (torch.sigmoid(pconf / betasig) - 0.5))[:, 0]
            for j, (sy, sx) in enumerate(chunk):
                accu_pred[..., sy, sx] += (pred[j * B:(j + 1) * B] * conf[j * B:(j + 1) * B, None])
                accu_conf[..., sy, sx] += conf[j * B:(j + 1) * B]
        pred = accu_pred / accu_conf[:, None]
        assert torch.isfinite(pred).all()
        return pred


def num_tiles(H, W, crop=(352, 704), overlap=0.7):
    return len(list(_overlapping(H, crop[0], overlap))) * len(list(_overlapping(W, crop[1], overlap)))

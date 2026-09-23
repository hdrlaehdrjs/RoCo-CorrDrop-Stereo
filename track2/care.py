"""CARE-Stereo: Correspondence-Aware Robust Epipolar Stereo modules for a frozen CroCo-Stereo E0.

The paper's final model uses only variant="dropout" (parameter-free structured correspondence dropout; the
random-key control uses the same path with an explicit key mask). The other variants are retained as development
utilities and are not used by the reported final model.

CroCo facts this relies on (third_party/croco, verified):
  * DecoderBlock.forward(x, y, xpos, ypos):
        x = x + attn(norm1(x), xpos)
        y_ = norm_y(y)
        x = x + cross_attn(norm2(x), y_, y_, xpos, ypos)      # CrossAttention: projq/k/v, RoPE on q,k, softmax, proj
        x = x + mlp(norm3(x))
        return x, y                                          # right tokens are never updated by the decoder
  * tokens are raster ordered (row-major); pos[..., 0] = token row, pos[..., 1] = token column (patch 16)
  * cuRoPE2D rotates IN PLACE -> global and epipolar q/k are computed from separate projections
  * the DPT head reads encoder block 23 and decoder blocks 3, 7, 11; CARE defaults to decoder block 7 so the
    correction reaches two of the four DPT inputs (blocks 7 and 11)

Frozen-backbone variants (installed on ONE decoder block; all original CroCo parameters stay frozen):
  adapter  : z_new = z + Up(GELU(Down(LN(z))))                  , Up zero-init
  epipolar : C_new = C_global + W_rob(C_epi - C_global)          , W_rob bottleneck, Up zero-init
  rg_eca   : C_new = C_global + W_rob(g * (C_epi - C_global)) , g = sigmoid(MLP(reliability stats))
with C_epi = proj(softmax_row((Q0 + dQ)(K0 + dK)^T / sqrt(d)) V0): same-row (epipolar) attention reusing the frozen
projections, low-rank zero-init dQ/dK adapters. Reliability statistics come from the frozen global cross-attention
only (entropy, epipolar mass, vertical spread, max prob, top1-top2 margin): no GT, no corruption label.
With the zero-initialised up-projections the patched block is numerically identical to CroCo E0
(tests/test_care.py).

Jointly trained variants (trained with the whole network, typically on every decoder block):
  dropout  : parameter-free. Correspondence dropout = for selected left queries (care.qdrop, training only) the keys
             within +-drop_band token rows of the query row are masked in the ORIGINAL cross-attention, so the true
             match is unavailable and the twin losses must be met from context. Identity without a mask.
  fallback : dropout + reliability-gated fallback on the original cross-attention output
                 y = g * CA(x) + (1 - g) * m,   g = sigmoid(MLP(attention statistics) + b)
             m = learned "no-correspondence" embedding (zero-init), MLP output layer zero-init, b = fallback_bias_init
             (sigmoid(5) = 0.993: near-identity at init). Statistics are GT-free (inference-available).
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CAREConfig:
    variant: str = "rg_eca"      # none | adapter | epipolar | rg_eca | probe | dropout | fallback
    block: int = 7               # decoder block index
    qk_rank: int = 8             # low-rank Q/K adapter rank
    bottleneck: int = 64         # W_rob / generic adapter bottleneck
    eps_token: int = 0           # epipolar row tolerance in tokens (0 = exact row; >0 uses a dense masked path)
    gate_hidden: int = 16
    gate_bias_init: float = -2.0  # sigmoid(-2) = 0.12: gate starts mostly closed
    force_dense: bool = False    # debug/test: masked dense epipolar attention even for eps_token = 0
    blocks: list | None = None   # v2: several decoder blocks (e.g. [7, 11]); None -> [block]
    gate_features: str = "v1"    # v1: 5 frozen global-attention statistics; v2: + 6 global/epipolar agreement statistics
    fallback_bias_init: float = 5.0  # fallback: sigmoid(5) = 0.993, gate starts open (near-identity)
    drop_band: int = 1           # dropout/fallback: key rows |dy| <= drop_band are masked for dropped queries

    def to_dict(self):
        return asdict(self)


class LowRank(nn.Module):
    def __init__(self, dim, rank, out=None, act=False, norm=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if norm else nn.Identity()
        self.down = nn.Linear(dim, rank, bias=act)
        self.act = nn.GELU() if act else nn.Identity()
        self.up = nn.Linear(rank, out or dim, bias=act)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(self.norm(x))))


def attention_stats(A: torch.Tensor, Ht: int, Wt: int, eps_token: int = 0) -> torch.Tensor:
    """A: head-averaged global cross-attention B x N x N (rows sum to 1). Returns B x N x 5:
    [normalised entropy, epipolar mass, vertical spread, max prob, top1-top2 margin], all in [0, 1]."""
    B, N, _ = A.shape
    Af = A.float()
    ent = -(Af.clamp_min(1e-12).log() * Af).sum(-1) / math.log(N)
    rows = torch.arange(Ht, device=A.device).repeat_interleave(Wt)                     # N
    dy = (rows[None, :] - rows[:, None]).abs()                                          # N x N
    band = (dy <= eps_token).float()
    m_epi = (Af * band[None]).sum(-1)
    v_y = (Af * ((dy.float() / max(1, Ht - 1)) ** 2)[None]).sum(-1)
    top2 = Af.topk(2, dim=-1).values
    feats = torch.stack([ent, m_epi, v_y, top2[..., 0], top2[..., 0] - top2[..., 1]], -1)
    return feats.clamp(0, 1)


def agreement_stats(glob_row: torch.Tensor, epi_row: torch.Tensor) -> torch.Tensor:
    """Row distributions B x Ht x Wt x Wt (global restricted to the epipolar row, and epipolar branch).
    Returns B x Ht x Wt x 6 in [0, 1]: [peak global, peak epipolar, JS(global, epi)/log 2,
    |argmax_g - argmax_e| / Wt, entropy global / log Wt, entropy epi / log Wt]. GT-free (inference-available)."""
    Wt = glob_row.shape[-1]
    P = glob_row.float().clamp_min(1e-12)
    Q = epi_row.float().clamp_min(1e-12)
    M = 0.5 * (P + Q)
    js = 0.5 * (P * (P.log() - M.log())).sum(-1) + 0.5 * (Q * (Q.log() - M.log())).sum(-1)
    dist = (P.argmax(-1) - Q.argmax(-1)).abs().float() / max(1, Wt - 1)
    ent = lambda A: -(A * A.log()).sum(-1) / math.log(Wt)
    return torch.stack([P.amax(-1), Q.amax(-1), js / math.log(2), dist, ent(P), ent(Q)], -1).clamp(0, 1)


class CAREBlockModule(nn.Module):
    """Trainable CARE parameters for one CroCo DecoderBlock of width `dim` with `heads` heads."""

    def __init__(self, cfg: CAREConfig, dim: int, heads: int):
        super().__init__()
        self.cfg = cfg
        self.dim, self.heads = dim, heads
        if cfg.variant == "adapter":
            self.adapter = LowRank(dim, cfg.bottleneck, act=True, norm=True)
        if cfg.variant in ("epipolar", "rg_eca"):
            self.dq = LowRank(dim, cfg.qk_rank)
            self.dk = LowRank(dim, cfg.qk_rank)
            self.w_rob = LowRank(dim, cfg.bottleneck, act=True)
        if cfg.variant == "rg_eca":
            n_in = 5 if cfg.gate_features == "v1" else 11
            self.gate = nn.Sequential(nn.Linear(n_in, cfg.gate_hidden), nn.GELU(), nn.Linear(cfg.gate_hidden, 1))
            nn.init.constant_(self.gate[-1].bias, cfg.gate_bias_init)
        if cfg.variant == "fallback":
            self.gate = nn.Sequential(nn.Linear(5, cfg.gate_hidden), nn.GELU(), nn.Linear(cfg.gate_hidden, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, cfg.fallback_bias_init)
            self.m = nn.Parameter(torch.zeros(dim))          # no-correspondence embedding
        self.enabled = True
        self.record = False
        self.cache: dict = {}
        self.qdrop = None      # B x N bool, training only: queries whose epipolar key band is masked (set by care_keydrop)
        self.kmask = None      # B x N x N bool, optional explicit key mask replacing the epipolar band (random-key control)
        self.last_g = None     # B x N, fallback: gate of the last forward (detached; logged by the trainer)


def _heads(t, B, N, h):
    return t.reshape(B, N, h, t.shape[-1] // h).permute(0, 2, 1, 3)


def care_block_forward(block, care: CAREBlockModule, x, y, xpos, ypos):
    """Drop-in replacement of DecoderBlock.forward (identical when care.enabled is False or residuals are zero)."""
    x = x + block.drop_path(block.attn(block.norm1(x), xpos))
    y_ = block.norm_y(y)
    xn = block.norm2(x)
    cfg = care.cfg
    ca = block.cross_attn
    if cfg.variant == "probe" and care.enabled and care.record:
        # bit-identical re-implementation of CrossAttention.forward that keeps the head-averaged attention (with grad)
        B, N, C = xn.shape
        h = ca.num_heads
        q = _heads(ca.projq(xn), B, N, h)
        k = _heads(ca.projk(y_), B, y_.shape[1], h)
        v = _heads(ca.projv(y_), B, y_.shape[1], h)
        if ca.rope is not None:
            q = ca.rope(q, xpos)
            k = ca.rope(k, ypos)
        attn = ca.attn_drop(((q @ k.transpose(-2, -1)) * ca.scale).softmax(dim=-1))
        x = x + block.drop_path(ca.proj_drop(ca.proj((attn @ v).transpose(1, 2).reshape(B, N, C))))
        x = x + block.drop_path(block.mlp(block.norm3(x)))
        Ht = int(xpos[0, :, 0].max()) + 1
        care.cache.clear()
        care.cache.update(attn_full=attn.mean(1), grid=(Ht, N // Ht))
        return x, y
    if cfg.variant in ("dropout", "fallback"):
        if not care.enabled or (cfg.variant == "dropout" and care.qdrop is None and not care.record):
            x = x + block.drop_path(ca(xn, y_, y_, xpos, ypos))
            x = x + block.drop_path(block.mlp(block.norm3(x)))
            return x, y
        # original cross-attention re-implemented (bit-identical) so that the key band can be masked and the gate
        # can read the attention statistics
        B, N, C = xn.shape
        Nk = y_.shape[1]
        h = ca.num_heads
        Ht = int(xpos[0, :, 0].max()) + 1
        Wt = N // Ht
        q = _heads(ca.projq(xn), B, N, h)
        k = _heads(ca.projk(y_), B, Nk, h)
        v = _heads(ca.projv(y_), B, Nk, h)
        if ca.rope is not None:
            q = ca.rope(q, xpos)
            k = ca.rope(k, ypos)
        logits = (q @ k.transpose(-2, -1)) * ca.scale
        if care.qdrop is not None:
            assert Nk == N, "correspondence dropout expects same-size left/right token grids"
            km = care.kmask if care.kmask is not None else drop_mask(care.qdrop, Ht, Wt, cfg.drop_band)
            logits = logits.masked_fill(km[:, None], float("-inf"))
        attn = ca.attn_drop(logits.softmax(dim=-1))
        c_out = ca.proj_drop(ca.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))
        g = stats = None
        if cfg.variant == "fallback":
            with torch.no_grad():
                stats = attention_stats(attn.detach().mean(1), Ht, Wt, cfg.eps_token)
            g = torch.sigmoid(care.gate(stats.to(xn.dtype)))                                  # B N 1
            care.last_g = g.detach().reshape(B, N)
            c_out = g * c_out + (1 - g) * care.m.to(xn.dtype)
        x = x + block.drop_path(c_out)
        x = x + block.drop_path(block.mlp(block.norm3(x)))
        if care.record:
            care.cache.clear()
            care.cache.update(attn_full=attn.mean(1), grid=(Ht, Wt), stats=stats,
                              g=None if g is None else g.detach().reshape(B, Ht, Wt))
        return x, y
    if not care.enabled or cfg.variant in ("none", "adapter", "probe"):
        x = x + block.drop_path(ca(xn, y_, y_, xpos, ypos))
        x = x + block.drop_path(block.mlp(block.norm3(x)))
        if care.enabled and cfg.variant == "adapter":
            x = x + care.adapter(x)
        return x, y

    B, N, C = xn.shape
    Nk = y_.shape[1]
    h = ca.num_heads
    Ht = int(xpos[0, :, 0].max()) + 1
    Wt = N // Ht
    assert Ht * Wt == N and Nk == N, "CARE expects same-size left/right token grids"
    # ---- frozen global cross-attention, bit-identical to CrossAttention.forward
    q = _heads(ca.projq(xn), B, N, h)
    k = _heads(ca.projk(y_), B, Nk, h)
    v = _heads(ca.projv(y_), B, Nk, h)
    if ca.rope is not None:
        q = ca.rope(q, xpos)
        k = ca.rope(k, ypos)
    attn = ((q @ k.transpose(-2, -1)) * ca.scale).softmax(dim=-1)
    attn = ca.attn_drop(attn)
    c_global = ca.proj_drop(ca.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))
    # ---- epipolar branch (same-row attention, residual Q/K adapters, frozen V and output projection)
    qe = _heads(ca.projq(xn) + care.dq(xn), B, N, h)
    ke = _heads(ca.projk(y_) + care.dk(y_), B, Nk, h)
    if ca.rope is not None:
        qe = ca.rope(qe, xpos)
        ke = ca.rope(ke, ypos)
    d = C // h
    if cfg.eps_token == 0 and not cfg.force_dense:
        qr = qe.reshape(B, h, Ht, Wt, d)
        kr = ke.reshape(B, h, Ht, Wt, d)
        vr = v.reshape(B, h, Ht, Wt, d)
        a_epi = ((qr @ kr.transpose(-2, -1)) * ca.scale).softmax(dim=-1)              # B h Ht Wt Wt
        out = (a_epi @ vr).reshape(B, h, N, d)
    else:
        rows = torch.arange(Ht, device=xn.device).repeat_interleave(Wt)
        mask = (rows[None, :] - rows[:, None]).abs() > cfg.eps_token
        logits = ((qe @ ke.transpose(-2, -1)) * ca.scale).masked_fill(mask, float("-inf"))
        a_epi_dense = logits.softmax(dim=-1)
        out = a_epi_dense @ v
        a_epi = a_epi_dense.reshape(B, h, Ht, Wt, Ht, Wt)[:, :, torch.arange(Ht), :, torch.arange(Ht), :].permute(1, 2, 0, 3, 4) \
            if cfg.eps_token == 0 else None
    c_epi = ca.proj_drop(ca.proj(out.transpose(1, 2).reshape(B, N, C)))
    glob_row = a_epi_m = None
    if care.record or (cfg.variant == "rg_eca" and cfg.gate_features == "v2"):
        assert a_epi is not None, "row distributions are defined on the exact epipolar row (eps_token=0)"
        # same-row global attention, renormalised PER HEAD then head-averaged: exactly the normalisation of A_epi,
        # so A_epi == A_global_epi when dQ = dK = 0 and any difference is learned (mechanism diagnostic Q9)
        idx = torch.arange(Ht, device=xn.device)
        glob_row = attn.reshape(B, h, Ht, Wt, Ht, Wt)[:, :, idx, :, idx, :].permute(1, 2, 0, 3, 4)  # B h Ht Wt Wt
        glob_row = (glob_row / glob_row.sum(-1, keepdim=True).clamp_min(1e-12)).mean(1)
        a_epi_m = a_epi.mean(1)
    # ---- reliability gate (inference-available inputs only: no GT, no corruption label)
    if cfg.variant == "rg_eca":
        with torch.no_grad():
            abar = attn.detach().mean(1)
            stats = attention_stats(abar, Ht, Wt, cfg.eps_token)
            if cfg.gate_features == "v2":
                stats = torch.cat([stats, agreement_stats(glob_row.detach(), a_epi_m.detach()).reshape(B, N, 6)], -1)
        g = torch.sigmoid(care.gate(stats.to(xn.dtype)))                                  # B N 1
    else:
        g = torch.ones(B, N, 1, device=xn.device, dtype=xn.dtype)
        stats = None
    delta = care.w_rob(g * (c_epi - c_global))
    x = x + block.drop_path(c_global + delta)
    x = x + block.drop_path(block.mlp(block.norm3(x)))
    if care.record:
        gg = g.reshape(B, Ht, Wt, 1)
        a_eff = (1 - gg) * glob_row + gg * a_epi_m
        a_eff = a_eff / a_eff.sum(-1, keepdim=True).clamp_min(1e-12)
        care.cache.clear()  # update in place: care_recording() hands out this dict object
        care.cache.update(g=g.reshape(B, Ht, Wt), a_eff=a_eff, c_epi=c_epi.detach(), c_global=c_global.detach(), a_glob_row=glob_row, a_epi=a_epi_m, stats=stats,
                          delta_norm=delta.detach().norm(dim=-1).mean(), grid=(Ht, Wt))
    return x, y


class CAREMulti(nn.Module):
    """CARE on several decoder blocks. state_dict prefix 'care.blocks.b<k>.'; cache = {k: per-block cache}."""

    def __init__(self, cfg: CAREConfig, children: dict):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleDict({f"b{k}": m for k, m in children.items()})
        self.cache = {k: m.cache for k, m in children.items()}

    def _set(self, name, v):
        for m in self.blocks.values():
            setattr(m, name, v)

    enabled = property(lambda self: next(iter(self.blocks.values())).enabled, lambda self, v: self._set("enabled", v))
    record = property(lambda self: next(iter(self.blocks.values())).record, lambda self, v: self._set("record", v))
    qdrop = property(lambda self: next(iter(self.blocks.values())).qdrop, lambda self, v: self._set("qdrop", v))


def _care_modules(net) -> list:
    care = getattr(net, "care", None)
    if care is None:
        return []
    return [care] if isinstance(care, CAREBlockModule) else list(care.blocks.values())


_BAND_CACHE: dict = {}


def drop_mask(qdrop: torch.Tensor, Ht: int, Wt: int, band: int) -> torch.Tensor:
    """qdrop B x N bool (queries whose true correspondence is hidden) -> B x N x N bool logits mask: for every dropped
    query, all right tokens within `band` token rows of its own row (the whole epipolar band)."""
    key = (Ht, Wt, band, str(qdrop.device))
    if key not in _BAND_CACHE:
        rows = torch.arange(Ht, device=qdrop.device).repeat_interleave(Wt)
        _BAND_CACHE[key] = (rows[None, :] - rows[:, None]).abs() <= band
    return qdrop[:, :, None] & _BAND_CACHE[key][None]


@torch.no_grad()
def sample_query_drop(seeds, grid: tuple[int, int], p: float, area=(0.05, 0.3)) -> torch.Tensor:
    """Per-sample rectangular query-drop region on the Ht x Wt left-token grid (deterministic in the seed): with
    probability p a box covering U(area) of the grid, aspect ratio within e^+-0.7 of the grid's. Returns B x N bool."""
    Ht, Wt = grid
    out = torch.zeros(len(seeds), Ht, Wt, dtype=torch.bool)
    for b, s in enumerate(seeds):
        gen = torch.Generator().manual_seed(int(s) + 104729)
        u = torch.rand(3, generator=gen).tolist()
        if u[0] >= p:
            continue
        n = (area[0] + (area[1] - area[0]) * u[1]) * Ht * Wt
        aspect = math.exp((u[2] * 2 - 1) * 0.7)
        ww = int(round(max(1, min(Wt, math.sqrt(n / (aspect * Ht / Wt))))))
        hh = int(round(max(1, min(Ht, n / ww))))
        y0 = int(torch.randint(0, Ht - hh + 1, (1,), generator=gen))
        x0 = int(torch.randint(0, Wt - ww + 1, (1,), generator=gen))
        out[b, y0:y0 + hh, x0:x0 + ww] = True
    return out.reshape(len(seeds), Ht * Wt)


@torch.no_grad()
def sample_random_key_mask(qdrop: torch.Tensor, seeds, grid: tuple[int, int], band: int) -> torch.Tensor:
    """Budget-matched random-key CONTROL of correspondence dropout. For every dropped query, exactly as many right
    keys as its epipolar band would hide (Wt * #rows within `band` of the query row: 132 for an interior row of the
    22 x 44 grid, 88 for the first / last row) are drawn uniformly WITHOUT replacement from all N keys. One draw per
    sample and forward, shared by all heads and decoder layers like the structured mask; deterministic in the sample
    seed (which derives from the run seed). Returns B x N x N bool on qdrop's device."""
    Ht, Wt = grid
    N = Ht * Wt
    rows = torch.arange(Ht).repeat_interleave(Wt)
    budget = ((rows[None, :] - rows[:, None]).abs() <= band).sum(1)                       # N, keys hidden per query
    q = qdrop.cpu()
    out = torch.zeros(len(seeds), N, N, dtype=torch.bool)
    for b, s in enumerate(seeds):
        if not bool(q[b].any()):
            continue
        gen = torch.Generator().manual_seed(int(s) + 1299709)
        rank = torch.rand(N, N, generator=gen).argsort(1).argsort(1)                      # random permutation per query
        out[b] = (rank < budget[:, None]) & q[b][:, None]
    return out.to(qdrop.device)


@contextmanager
def care_keydrop(net, qdrop: torch.Tensor | None, kmask: torch.Tensor | None = None):
    """Correspondence dropout for one forward: every dropout/fallback block masks the epipolar key band of qdrop
    (or the explicit B x N x N key mask `kmask`, used by the random-key control)."""
    mods = _care_modules(net) if qdrop is not None else []
    for m in mods:
        m.qdrop = qdrop
        m.kmask = kmask
    try:
        yield
    finally:
        for m in mods:
            m.qdrop = None
            m.kmask = None


def care_gate_values(net) -> list:
    """Detached B x N gates of the last forward, one per fallback block (empty for other variants)."""
    return [m.last_g for m in _care_modules(net) if m.cfg.variant == "fallback" and m.last_g is not None]


def care_block_items(cache: dict) -> list:
    """[(block index or None, per-block cache)] from a care_recording() dict (single- or multi-block)."""
    if not cache:
        return []
    if "grid" in cache or "g" in cache or "attn_full" in cache:
        return [(None, cache)]
    return [(k, cache[k]) for k in sorted(cache)]


def install_care(net: nn.Module, cfg: CAREConfig):
    """Attach CARE to net.dec_blocks[cfg.block] (or every block in cfg.blocks). Original parameter names are unchanged;
    CARE parameters live in net.care (state_dict prefix 'care.'), so plain E0 checkpoints still load strictly."""
    if cfg.variant == "none":
        return None
    blocks = list(cfg.blocks) if cfg.blocks else [cfg.block]

    def attach(k, bcfg):
        block = net.dec_blocks[k]
        mod = CAREBlockModule(bcfg, net.dec_embed_dim, block.cross_attn.num_heads)
        block.forward = lambda x, y, xpos, ypos: care_block_forward(block, mod, x, y, xpos, ypos)
        return mod

    if len(blocks) == 1 and not cfg.blocks:
        care = attach(cfg.block, cfg)
    else:
        from dataclasses import replace
        care = CAREMulti(cfg, {k: attach(k, replace(cfg, block=k)) for k in blocks})
    net.care = care
    return care


def care_block_caches(cache: dict) -> list:
    """Per-block caches from a care_recording() dict (single- or multi-block)."""
    if not cache:
        return []
    if "grid" in cache or "g" in cache or "attn_full" in cache:
        return [cache]
    return [cache[k] for k in sorted(cache)]


def snapshot_cache(cache: dict) -> list:
    return [dict(c) for c in care_block_caches(cache)]


def first_care_block(net) -> int:
    care = net.care
    return min(care.cfg.blocks) if getattr(care.cfg, "blocks", None) else care.cfg.block


@contextmanager
def care_disabled(net):
    care = getattr(net, "care", None)
    if care is None:
        yield
        return
    old = care.enabled
    care.enabled = False
    try:
        yield
    finally:
        care.enabled = old


@contextmanager
def care_recording(net):
    care = getattr(net, "care", None)
    if care is None:
        yield {}
        return
    care.record = True
    for c in ([care.cache] if isinstance(care, CAREBlockModule) else list(care.cache.values())):
        c.clear()
    try:
        yield care.cache
    finally:
        care.record = False


# ------------------------------------------------------------------ correspondence supervision
@torch.no_grad()
def token_targets(gt: torch.Tensor, grid: tuple[int, int], patch: int = 16, radius: float = 1.5,
                  min_valid: float = 0.5, max_spread: float = 4.0):
    """GT right-token distribution per left token, on the same row.

    gt: B x 1 x H x W disparity in CROP pixels (after all augmentation resizes, so no extra scaling).
    For left token (i, j) with centre x_c = 16 j + 7.5: d = mean valid GT inside the patch (requires >= min_valid
    valid pixels and std <= max_spread px, i.e. no depth edge), x_R* = x_c - d, u* = (x_R* - 7.5) / 16 in right-token
    columns. Target = triangular kernel max(0, 1 - |u - u*| / radius), normalised; invalid if u* outside the row.
    Returns T (B Ht Wt Wt) and valid (B Ht Wt)."""
    Ht, Wt = grid
    B = gt.shape[0]
    g = gt[:, 0, :Ht * patch, :Wt * patch]
    fin = torch.isfinite(g) & (g >= 0)
    gz = torch.where(fin, g, torch.zeros_like(g)).reshape(B, Ht, patch, Wt, patch).permute(0, 1, 3, 2, 4).reshape(B, Ht, Wt, -1)
    fz = fin.reshape(B, Ht, patch, Wt, patch).permute(0, 1, 3, 2, 4).reshape(B, Ht, Wt, -1).float()
    n = fz.sum(-1)
    mean = gz.sum(-1) / n.clamp_min(1)
    var = ((gz - mean[..., None]) ** 2 * fz).sum(-1) / n.clamp_min(1)
    xc = torch.arange(Wt, device=gt.device, dtype=g.dtype) * patch + (patch - 1) / 2
    u = (xc[None, None, :] - mean - (patch - 1) / 2) / patch                                 # B Ht Wt
    valid = (n >= min_valid * patch * patch) & (var.sqrt() <= max_spread) & (u >= -0.5) & (u <= Wt - 0.5)
    cols = torch.arange(Wt, device=gt.device, dtype=g.dtype)
    T = (1 - (cols[None, None, None, :] - u[..., None]).abs() / radius).clamp_min(0)
    T = T / T.sum(-1, keepdim=True).clamp_min(1e-12)
    T = torch.where(valid[..., None], T, torch.zeros_like(T))
    return T, valid


def epi_loss(a_eff: torch.Tensor, T: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6):
    ce = -(T * (a_eff.float() + eps).log()).sum(-1)
    return ce[valid].mean() if valid.any() else a_eff.sum() * 0


def attn_consistency(a_clean: torch.Tensor, a_corr: torch.Tensor, valid: torch.Tensor, mode: str = "kl", eps: float = 1e-6):
    """KL(P || Q) with P = sg(A_eff_clean), Q = A_eff_corr over valid queries (JS as documented fallback)."""
    P = a_clean.detach().float().clamp_min(eps)
    Q = a_corr.float().clamp_min(eps)
    P, Q = P / P.sum(-1, keepdim=True), Q / Q.sum(-1, keepdim=True)
    if mode == "kl":
        d = (P * (P.log() - Q.log())).sum(-1)
    else:
        M = 0.5 * (P + Q)
        d = 0.5 * (P * (P.log() - M.log())).sum(-1) + 0.5 * (Q * (Q.log() - M.log())).sum(-1)
    return d[valid].mean() if valid.any() else a_corr.sum() * 0


@torch.no_grad()
def gt_mass(A: torch.Tensor, T: torch.Tensor, valid: torch.Tensor) -> float:
    """Attention mass on the GT support (T > 0), averaged over valid queries."""
    m = (A.float() * (T > 0)).sum(-1)
    return float(m[valid].mean()) if valid.any() else float("nan")


def care_parameters(net):
    care = getattr(net, "care", None)
    return [] if care is None else list(care.parameters())


# ------------------------------------------------------------------ attention-level correspondence losses (direction 1)
def full_attention_epi_loss(attn_full: torch.Tensor, T: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6):
    """Cross-entropy of the GT right-token distribution T (B Ht Wt Wt, same row) under the ORIGINAL head-averaged
    cross-attention over ALL right tokens (B N N, not renormalised): pulls attention mass onto the GT correspondence
    on the epipolar row."""
    B, N, _ = attn_full.shape
    Ht, Wt = T.shape[1], T.shape[2]
    idx = torch.arange(Ht, device=attn_full.device)
    row = attn_full.reshape(B, Ht, Wt, Ht, Wt)[:, idx, :, idx, :].permute(1, 0, 2, 3)          # B Ht Wt Wt
    ce = -(T * (row.float() + eps).log()).sum(-1)
    return ce[valid].mean() if valid.any() else attn_full.sum() * 0


def full_attention_consistency(a_clean: torch.Tensor, a_corr: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6):
    """KL(sg(A_clean) || A_corr) over all right tokens for valid left tokens (B N N, head-averaged)."""
    B, N, _ = a_clean.shape
    P = a_clean.detach().float().clamp_min(eps)
    Q = a_corr.float().clamp_min(eps)
    d = (P * (P.log() - Q.log())).sum(-1)                                                       # B N
    v = valid.reshape(B, N)
    return d[v].mean() if v.any() else a_corr.sum() * 0

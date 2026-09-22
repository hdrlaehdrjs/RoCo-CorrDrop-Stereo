"""CARE-Stereo unit tests on a tiny CPU CroCo-Stereo (cosine positional embedding).
The real E0 (RoPE, ViT-L) parity is checked on GPU by scripts/care_parity_gpu.py."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__import__("os").environ.get("ROCO_TRACK2_WORK", ROOT))   # checkpoints/, outputs/, third_party/croco live here
sys.path[:0] = [str(ROOT), str(WORK / "third_party/croco")]
from models.croco_downstream import CroCoDownstreamBinocular  # noqa: E402
from models.head_downstream import PixelwiseTaskWithDPT  # noqa: E402
from track2.care import (full_attention_epi_loss, full_attention_consistency, agreement_stats, snapshot_cache, CAREConfig, install_care, care_disabled, care_recording, attention_stats, token_targets,  # noqa: E402
                         epi_loss, attn_consistency, care_parameters, care_keydrop, care_block_items, sample_query_drop)
from track2.croco_model import Track2CroCoStereo, build_croco_stereo  # noqa: E402
from track2.cvcaug import CVCAugConfig, cvcaug  # noqa: E402

H, W = 64, 128  # 4 x 8 tokens
TINY_ARGS = dict(img_size=(H, W), patch_size=16, enc_embed_dim=64, enc_depth=2, enc_num_heads=4,
                 dec_embed_dim=48, dec_num_heads=4, dec_depth=8, pos_embed="cosine")


def tiny(seed=0):
    torch.manual_seed(seed)
    head = PixelwiseTaskWithDPT()
    head.num_channels = 2
    net = CroCoDownstreamBinocular(head, **TINY_ARGS)
    net.eval()
    return net


def pair(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, H, W, generator=g), torch.randn(2, 3, H, W, generator=g)


def randomize_care(net):
    with torch.no_grad():
        for p in care_parameters(net):
            p.add_(torch.randn_like(p) * 0.05)


@pytest.mark.parametrize("variant", ["adapter", "epipolar", "rg_eca"])
def test_1_parity_and_7_zero_residual(variant):
    net = tiny()
    l, r = pair()
    with torch.no_grad():
        ref = net(l, r)
    install_care(net, CAREConfig(variant=variant, block=5))
    with torch.no_grad(), care_recording(net) as cache:
        new = net(l, r)
    assert (ref - new).abs().max() < 1e-5
    if variant != "adapter":
        assert float(cache["delta_norm"]) == 0.0
    with torch.no_grad(), care_disabled(net):
        assert (net(l, r) - ref).abs().max() == 0


def test_frozen_prefix_decode_matches_decode():
    net = tiny()
    install_care(net, CAREConfig(block=5))
    randomize_care(net)
    m = Track2CroCoStereo(net, dict(crop=[H, W], croco_args={}, criterion="", tile_conf_mode="conf_expsigmoid_15_3"))
    l, r = pair()
    with torch.no_grad():
        enc = m.encode(l, r)
        a = m.decode(enc, l.shape)
        b = m.decode_frozen_prefix(enc, l.shape, 5)
    assert all((x - y).abs().max() < 1e-5 for x, y in zip(a, b))


def test_2_epipolar_mask_fast_equals_dense_and_rows_only():
    net = tiny()
    care = install_care(net, CAREConfig(variant="epipolar", block=5))
    randomize_care(net)
    l, r = pair()
    with torch.no_grad(), care_recording(net) as c:
        net(l, r)
        fast = c["c_epi"].clone()
    care.cfg.force_dense = True
    with torch.no_grad(), care_recording(net) as c:
        net(l, r)
        dense = c["c_epi"].clone()
    assert (fast - dense).abs().max() < 1e-4
    # changing right tokens of other rows must not change the epipolar output of row 0
    care.cfg.force_dense = False
    captured = {}
    blk = net.dec_blocks[5]
    orig = blk.forward

    def fwd(x, y, xpos, ypos):
        y2 = y.clone()
        y2[:, 8:] = torch.randn_like(y2[:, 8:])  # rows 1..3 of the 4x8 right grid (LayerNorm removes constant shifts)
        captured["y2"] = y2
        return orig(x, y2, xpos, ypos)
    with torch.no_grad(), care_recording(net) as c:
        blk.forward = fwd
        net(l, r)
        pert = c["c_epi"].clone()
    blk.forward = orig
    assert (pert[:, :8] - fast[:, :8]).abs().max() < 1e-5
    assert (pert[:, 8:] - fast[:, 8:]).abs().max() > 1e-3


def test_3_4_gt_token_targets_constant_disparity():
    gt = torch.full((1, 1, 64, 128), 32.0)
    T, v = token_targets(gt, (4, 8))
    assert not v[0, :, :2].any() and v[0, :, 2:].all()
    assert torch.equal(T[0, :, 2:].argmax(-1), torch.arange(0, 6).expand(4, 6))
    assert torch.allclose(T[v].sum(-1), torch.ones(int(v.sum())))
    gt2 = gt.clone()
    gt2[:, :, :16, 16:32] = float("nan")          # token (0,1) fully invalid
    gt2[:, :, 16:32, 48:56] = 0.0                 # token (1,3) straddles a depth edge
    _, v2 = token_targets(gt2, (4, 8))
    assert not v2[0, 0, 1] and not v2[0, 1, 3]


def test_5_attention_stats():
    N, Ht, Wt = 32, 4, 8
    uni = torch.full((1, N, N), 1.0 / N)
    one = torch.zeros(1, N, N)
    one[0, torch.arange(N), torch.arange(N)] = 1.0
    su, so = attention_stats(uni, Ht, Wt), attention_stats(one, Ht, Wt)
    assert torch.isfinite(su).all() and (su >= 0).all() and (su <= 1).all()
    assert abs(float(su[..., 0].mean()) - 1) < 1e-5 and float(so[..., 0].max()) < 1e-5
    assert torch.allclose(so[..., 1], torch.ones(1, N)) and abs(float(su[..., 1].mean()) - 0.25) < 1e-5
    assert torch.allclose(so[..., 3], torch.ones(1, N)) and torch.allclose(so[..., 4], torch.ones(1, N))


def test_6_gate_range():
    net = tiny()
    install_care(net, CAREConfig(variant="rg_eca", block=5))
    randomize_care(net)
    l, r = pair()
    with torch.no_grad(), care_recording(net) as c:
        net(l, r)
    assert (c["g"] >= 0).all() and (c["g"] <= 1).all() and torch.isfinite(c["a_eff"]).all()
    assert torch.allclose(c["a_eff"].sum(-1), torch.ones_like(c["a_eff"].sum(-1)), atol=1e-4)


def _imgs(h=96, w=192, seed=0):
    rs = np.random.default_rng(seed)
    return rs.integers(0, 255, (h, w, 3), dtype=np.uint8), rs.integers(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("mode", ["shared", "asym", "drop"])
def test_8_cvcaug_geometry_and_determinism(mode):
    L, R = _imgs()
    d = np.full((96, 192), 6.0, np.float32)
    d0 = d.copy()
    cfg = CVCAugConfig(p_aug=1.0, warmup_frac=0.0, p_shared=float(mode == "shared"), p_asym=float(mode == "asym"),
                       p_corr_dropout=float(mode == "drop"))
    for s in range(15):
        a = cvcaug(L, R, d, random.Random(s), np.random.default_rng(s), cfg)
        b = cvcaug(L, R, d, random.Random(s), np.random.default_rng(s), cfg)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
        assert a[0].shape == L.shape and a[1].shape == R.shape and np.array_equal(d, d0)
        assert a[2]["mode"] == {"shared": "shared", "asym": "asymmetric", "drop": "corr_dropout"}[mode]


def test_9_correspondence_dropout_transport_alignment():
    L = np.full((96, 192, 3), 128, np.uint8)
    R = L.copy()
    d = np.full((96, 192), 12.0, np.float32)
    cfg = CVCAugConfig(p_aug=1.0, warmup_frac=0.0, p_shared=0, p_asym=0, p_corr_dropout=1, p_recovery=0.0, n_boxes=(1, 1))
    for s in range(10):
        Lx, Rx, info = cvcaug(L, R, d, random.Random(s), np.random.default_rng(s), cfg)
        mL = np.abs(Lx.astype(int) - L).sum(-1) > 20
        mR = np.abs(Rx.astype(int) - R).sum(-1) > 20
        shifted = np.zeros_like(mL)
        shifted[:, :-12] = mL[:, 12:]
        inter, union = (shifted & mR).sum(), (shifted | mR).sum()
        assert info["recovery"] == 0 and union > 0 and inter / union > 0.8
    cfg.p_recovery = 1.0
    Lx, Rx, info = cvcaug(L, R, d, random.Random(3), np.random.default_rng(3), cfg)
    assert info["recovery"] == 1 and np.array_equal(Rx, R) and not np.array_equal(Lx, L)


def test_10_attention_consistency():
    g = torch.Generator().manual_seed(0)
    P = torch.softmax(torch.randn(2, 4, 8, 8, generator=g), -1)
    v = torch.ones(2, 4, 8, dtype=torch.bool)
    assert float(attn_consistency(P, P.clone(), v)) < 1e-6
    Q = torch.softmax(torch.randn(2, 4, 8, 8, generator=g), -1)
    assert float(attn_consistency(P, Q, v)) > 1e-3 and float(attn_consistency(P, Q, v, "js")) > 1e-4
    v0 = torch.zeros_like(v)
    assert float(attn_consistency(P, Q, v0)) == 0.0


def test_11_frozen_backbone_and_12_gradients():
    net = tiny()
    install_care(net, CAREConfig(variant="rg_eca", block=5))
    net.requires_grad_(False)
    net.care.requires_grad_(True)
    base = {n: p.detach().clone() for n, p in net.named_parameters() if not n.startswith("care.")}
    params = list(net.care.parameters())
    opt = torch.optim.AdamW(params, lr=1e-2)
    l, r = pair()
    gt = torch.full((2, 1, H, W), 16.0)
    target = torch.randn(2, 2, H, W)
    for it in range(4):
        opt.zero_grad()
        with care_recording(net) as c:
            out = net(l, r)
            T, v = token_targets(gt, c["grid"])
            loss = ((out - target) ** 2).mean() + epi_loss(c["a_eff"], T, v) + c["g"].mean()
        loss.backward()
        assert all(p.grad is None for n, p in net.named_parameters() if not n.startswith("care."))
        grads = {n: p.grad for n, p in net.care.named_parameters()}
        assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
        opt.step()
    assert all(float(g.abs().sum()) > 0 for g in grads.values()), [n for n, g in grads.items() if float(g.abs().sum()) == 0]
    assert all(torch.equal(p, base[n]) for n, p in net.named_parameters() if not n.startswith("care."))


def test_epipolar_equals_row_restricted_global_at_init():
    net = tiny()
    install_care(net, CAREConfig(variant="rg_eca", block=5))
    l, r = pair()
    with torch.no_grad(), care_recording(net) as c:
        net(l, r)
    assert (c["a_epi"] - c["a_glob_row"]).abs().max() < 1e-5
    assert torch.allclose(c["a_glob_row"].sum(-1), torch.ones_like(c["a_glob_row"].sum(-1)), atol=1e-4)


@pytest.mark.parametrize("blocks", [None, [5, 7]])
def test_v2_parity_multiblock_and_agreement_stats(blocks):
    net = tiny()
    l, r = pair()
    with torch.no_grad():
        ref = net(l, r)
    install_care(net, CAREConfig(variant="rg_eca", block=5, blocks=blocks, gate_features="v2"))
    with torch.no_grad(), care_recording(net) as cache:
        new = net(l, r)
        caches = snapshot_cache(cache)
    assert (ref - new).abs().max() < 1e-5
    assert len(caches) == (2 if blocks else 1)
    for c in caches:
        assert c["stats"].shape[-1] == 11 and (c["stats"] >= 0).all() and (c["stats"] <= 1).all()
    with torch.no_grad(), care_disabled(net):
        assert (net(l, r) - ref).abs().max() == 0
    sd = net.state_dict()
    assert all(not k.startswith("care.") or ("blocks.b" in k) == bool(blocks) for k in sd)


def test_agreement_stats_identical_and_disjoint():
    P = torch.zeros(1, 2, 8, 8); P[..., 3] = 1.0
    Q = torch.zeros(1, 2, 8, 8); Q[..., 6] = 1.0
    same, diff = agreement_stats(P, P), agreement_stats(P, Q)
    assert float(same[..., 2].max()) < 1e-4 and float(same[..., 3].max()) == 0.0
    assert float(diff[..., 2].min()) > 0.99 and abs(float(diff[..., 3].mean()) - 3 / 7) < 1e-5


@pytest.mark.parametrize("blocks", [None, [5, 7]])
def test_probe_variant_is_parameter_free_and_bit_identical(blocks):
    net = tiny()
    l, r = pair()
    with torch.no_grad():
        ref = net(l, r)
    n_params = sum(p.numel() for p in net.parameters())
    install_care(net, CAREConfig(variant="probe", block=5, blocks=blocks))
    assert sum(p.numel() for p in net.parameters()) == n_params
    with torch.no_grad(), care_recording(net) as cache:
        rec = net(l, r)
        cs = snapshot_cache(cache)
    with torch.no_grad():
        plain = net(l, r)
    assert (rec - ref).abs().max() < 1e-5 and (plain - ref).abs().max() == 0
    assert len(cs) == (2 if blocks else 1) and cs[0]["attn_full"].shape == (2, 32, 32)
    assert torch.allclose(cs[0]["attn_full"].sum(-1), torch.ones(2, 32), atol=1e-4)


def test_full_attention_losses():
    B, Ht, Wt = 1, 4, 8
    N = Ht * Wt
    gt = torch.full((B, 1, Ht * 16, Wt * 16), 16.0)
    T, v = token_targets(gt, (Ht, Wt))
    uni = torch.full((B, N, N), 1.0 / N, requires_grad=True)
    peaked = torch.full((B, N, N), 1e-6)
    for i in range(Ht):
        for j in range(Wt):
            peaked[0, i * Wt + j, i * Wt:(i + 1) * Wt] += T[0, i, j]   # attention equal to the GT row distribution
    peaked = peaked / peaked.sum(-1, keepdim=True)
    assert float(full_attention_epi_loss(peaked, T, v)) < float(full_attention_epi_loss(uni, T, v))
    full_attention_epi_loss(uni, T, v).backward()
    assert uni.grad is not None and torch.isfinite(uni.grad).all()
    assert float(full_attention_consistency(peaked, peaked.clone(), v)) < 1e-4
    assert float(full_attention_consistency(peaked, uni.detach(), v)) > 1.0


# ------------------------------------------------------------------ correspondence dropout / gated fallback
ALL8 = list(range(8))


def test_x6_dropout_variant_is_parameter_free_identity_and_masks_the_epipolar_band():
    net = tiny()
    l, r = pair()
    with torch.no_grad():
        ref = net(l, r)
    n_params = sum(p.numel() for p in net.parameters())
    install_care(net, CAREConfig(variant="dropout", block=0, blocks=ALL8, drop_band=1))
    assert sum(p.numel() for p in net.parameters()) == n_params
    with torch.no_grad():
        assert (net(l, r) - ref).abs().max() == 0                     # no mask, not recording: original path
    Ht, Wt = 4, 8
    qdrop = torch.zeros(2, Ht * Wt, dtype=torch.bool)
    qdrop[0, Wt:2 * Wt] = True                                        # sample 0, token row 1
    with torch.no_grad(), care_keydrop(net, qdrop), care_recording(net) as cache:
        out = net(l, r)
        items = care_block_items(cache)
    assert [k for k, _ in items] == ALL8
    rows = torch.arange(Ht).repeat_interleave(Wt)
    band = (rows[None] - rows[:, None]).abs() <= 1                    # rows 0..2 for a row-1 query
    for _, c in items:
        A = c["attn_full"]
        assert float(A[0][qdrop[0]][:, band[Wt]].sum()) == 0.0
        assert float(A[0][qdrop[0]][:, ~band[Wt]].min()) > 0.0
        assert torch.allclose(A.sum(-1), torch.ones(2, Ht * Wt), atol=1e-4)
    assert (out[1] - ref[1]).abs().max() < 1e-5                       # untouched sample
    assert (out[0] - ref[0]).abs().max() > 1e-4
    assert net.care.qdrop is None                                     # cleared after the context
    with torch.no_grad():
        assert (net(l, r) - ref).abs().max() == 0


def test_x6_fallback_near_identity_at_init_gate_open_and_trainable():
    net = tiny()
    l, r = pair()
    with torch.no_grad():
        ref = net(l, r)
    install_care(net, CAREConfig(variant="fallback", block=0, blocks=ALL8, fallback_bias_init=20.0))
    with torch.no_grad(), care_recording(net) as cache:
        out = net(l, r)
        items = care_block_items(cache)
    assert (out - ref).abs().max() < 1e-4                             # sigmoid(20) = 1 - 2e-9
    assert all(float(c["g"].min()) > 0.999 and c["stats"].shape[-1] == 5 for _, c in items)
    with torch.no_grad(), care_disabled(net):
        assert (net(l, r) - ref).abs().max() == 0
    net2 = tiny()
    install_care(net2, CAREConfig(variant="fallback", block=0, blocks=[3, 5]))
    assert len(list(net2.care.parameters())) == 2 * 5                 # per block: gate (4 tensors) + m
    assert all(float(p.abs().sum()) == 0 for n, p in net2.care.named_parameters() if n.endswith(".m") or "gate.2.weight" in n)
    with torch.no_grad(), care_recording(net2) as cache:
        net2(l, r)
        g = [c["g"] for _, c in care_block_items(cache)]
    assert all(abs(float(x.mean()) - 0.9933) < 1e-3 and x.shape == (2, 4, 8) for x in g)
    # a closed gate replaces the cross-attention output by m; gradients reach m and the gate output layer
    with torch.no_grad():
        for mod in net2.care.blocks.values():
            mod.gate[-1].bias.fill_(-20.0)
            mod.m.normal_()
    with care_recording(net2) as cache:
        out2 = net2(l, r)
        g = torch.stack([c["g"] for _, c in care_block_items(cache)])
    assert float(g.max()) < 1e-6 and (out2 - ref).abs().max() > 1e-3
    (out2.mean()).backward()
    grads = {n: p.grad for n, p in net2.care.named_parameters()}
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads.values())
    assert all(float(grads[n].abs().sum()) > 0 for n in grads if n.endswith(".m"))


def test_x6_fallback_checkpoint_roundtrip(tmp_path):
    net = tiny()
    install_care(net, CAREConfig(variant="fallback", block=0, blocks=[2, 5]))
    randomize_care(net)
    l, r = pair()
    with torch.no_grad():
        a = net(l, r)
    ck = tmp_path / "ck.pt"
    torch.save(dict(model=net.state_dict(), args=dict(croco_args=TINY_ARGS, crop=[H, W], criterion="LaplacianLossBounded2()",
                                                      tile_conf_mode="conf_expsigmoid_15_3", task="stereo"), care=net.care.cfg.to_dict()), ck)
    m, meta = build_croco_stereo(str(ck), "cpu")
    assert meta["care"]["variant"] == "fallback" and sorted(m.net.state_dict()) == sorted(net.state_dict())
    with torch.no_grad():
        b = m.net(l, r)
    assert (a - b).abs().max() < 1e-6
    # dropped queries also flow through the fallback path (mask applies to the gated attention)
    qdrop = torch.zeros(2, 32, dtype=torch.bool)
    qdrop[:, :8] = True
    with torch.no_grad(), care_keydrop(m.net, qdrop), care_recording(m.net) as cache:
        m.net(l, r)
        A = care_block_items(cache)[0][1]["attn_full"]
    assert float(A[:, :8][..., :16].sum()) == 0.0


def test_x6_sample_query_drop_deterministic_and_bounded():
    seeds = list(range(200))
    a = sample_query_drop(seeds, (22, 44), p=0.5, area=(0.05, 0.3))
    b = sample_query_drop(seeds, (22, 44), p=0.5, area=(0.05, 0.3))
    assert torch.equal(a, b) and a.shape == (200, 22 * 44)
    ev = a.any(1)
    assert 0.35 < float(ev.float().mean()) < 0.65
    frac = a[ev].float().mean(1)
    assert float(frac.min()) >= 0.03 and float(frac.max()) <= 0.36
    assert not sample_query_drop(seeds, (22, 44), p=0.0).any()
    assert sample_query_drop(seeds, (22, 44), p=1.0).any(1).all()


def test_random_key_control_is_budget_matched_random_seeded_and_masks_every_block():
    from track2.care import sample_random_key_mask, drop_mask
    # 1) budget: per dropped query exactly the number of keys the +-1-row epipolar band would hide (132 / 88 at borders)
    seeds = list(range(40))
    q = sample_query_drop(seeds, (22, 44), p=1.0, area=(0.05, 0.3))
    km = sample_random_key_mask(q, seeds, (22, 44), 1)
    band = drop_mask(q, 22, 44, 1)
    assert torch.equal(km.sum(-1), band.sum(-1))
    assert set(km.sum(-1)[q].tolist()) <= {88, 132} and not km[~q].any()
    # 2) genuinely random: overlap with the band ~ budget / N (13.6 %), far from 100 %; differs between samples
    overlap = float((km & band).sum(-1)[q].float().mean() / band.sum(-1)[q].float().mean())
    assert 0.10 < overlap < 0.17
    assert not torch.equal(km[0][q[0]][0], km[1][q[1]][0])
    # 3) seed-controlled: same seeds -> same mask, other seeds -> other mask; no global RNG consumed
    st = torch.get_rng_state()
    assert torch.equal(km, sample_random_key_mask(q, seeds, (22, 44), 1))
    assert torch.equal(st, torch.get_rng_state())
    assert not torch.equal(km[:20], sample_random_key_mask(q[:20], [s + 1000 for s in seeds[:20]], (22, 44), 1))
    # 4) the mask reaches the logits of every decoder block before the softmax
    net = tiny()
    l, r = pair()
    install_care(net, CAREConfig(variant="dropout", block=0, blocks=ALL8, drop_band=1))
    Ht, Wt = H // 16, W // 16
    qd = torch.zeros(2, Ht * Wt, dtype=torch.bool)
    qd[0, Wt:2 * Wt] = True
    k2 = sample_random_key_mask(qd, [5, 6], (Ht, Wt), 1)
    with torch.no_grad(), care_keydrop(net, qd, k2), care_recording(net) as cache:
        net(l, r)
        items = care_block_items(cache)
    assert [k for k, _ in items] == ALL8
    for _, c in items:
        A = c["attn_full"]
        assert float(A[k2].sum()) == 0.0 and float(A[0][qd[0]][~k2[0][qd[0]]].min()) > 0.0
        assert torch.allclose(A.sum(-1), torch.ones(2, Ht * Wt), atol=1e-4)
    assert net.care.qdrop is None and all(m.kmask is None for m in net.care.blocks.values())

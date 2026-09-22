#!/usr/bin/env python3
"""Twin-prediction robust training (backbone-agnostic).

Per step (memory-efficient, sequential backward):
  1. [no_grad] clean encoder features (shared by teacher and student when the student encoder is
     frozen and bit-identical to the teacher's; CroCo only)
  2. [no_grad] teacher clean prediction d_T (only if anchor / teacher-weighted delta / teacher or
     attention CorrMask need it)
  3. [no_grad] CorrMask placement on the corrupted pair (GPU), right-view transport with d_gt
  4. student clean forward:  L_clean + lambda_anchor * L_anchor  -> backward; keep sg(d_clean)
  5. student corrupted forward: lambda_corr * L_corr + lambda_delta * L_delta (+ lambda_feat L_feat)
     -> backward
  6. clip, optimizer step
Inference uses ONE student model (backbone.infer); the teacher never leaves this script.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__import__("os").environ.get("ROCO_TRACK2_WORK", ROOT))   # checkpoints/, outputs/, third_party/croco live here
sys.path.insert(0, str(ROOT))
from track2 import losses as Ls  # noqa: E402
from track2.backbones import make_backbone  # noqa: E402
from track2.config import load_config  # noqa: E402
from track2.s2aug import S2AugConfig  # noqa: E402
from track2.spring_data import load_split  # noqa: E402
from track2.stereo_corrmask import CorrMaskConfig, importance, sample_masks, apply_mask_events, mask_stats  # noqa: E402
from track2.twin_data import StepSampler, TwinStereoDataset  # noqa: E402
from contextlib import nullcontext  # noqa: E402
from track2.care import (care_recording, care_block_items, token_targets, full_attention_epi_loss,  # noqa: E402
                         full_attention_consistency, care_keydrop, sample_query_drop, sample_random_key_mask, drop_mask,
                         care_gate_values)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ramp(step, total, frac):
    return 1.0 if frac <= 0 else min(1.0, step / max(1.0, frac * total))


def lr_at(step, total, warmup, schedule):
    if step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * t))


def dataclass_from(cls, d):
    obj = cls()
    for k, v in (d or {}).items():
        if not hasattr(obj, k):
            raise KeyError(f"{cls.__name__} has no field {k}")
        setattr(obj, k, tuple(v) if isinstance(v, list) else v)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke/screening), schedule unchanged")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--overfit", type=int, default=0, help="repeat the first batch N times (loss must decrease)")
    ap.add_argument("--grad-probe", type=str, default="", help="start:end steps: log per-loss gradient norms over trainable params")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    L = cfg["loss"]
    s2cfg = dataclass_from(S2AugConfig, cfg.get("s2aug"))
    cmcfg = dataclass_from(CorrMaskConfig, {k: v for k, v in cfg.get("corrmask", {}).items() if k != "attn_blocks"})
    seed = cfg["seed"]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = a.out or (WORK / "outputs/track2" / cfg["name"] / f"seed{seed}_{stamp}")
    if out.exists():
        raise RuntimeError(f"{out} exists; never overwrite runs")
    out.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    bb = make_backbone(cfg)
    student = bb.build(cfg["init_checkpoint"])
    trainable = bb.configure_trainable(student, cfg["trainable"])
    need_teacher = (L.get("lambda_anchor", 0) > 0 or (L.get("lambda_delta", 0) > 0 and L.get("delta_weight") == "teacher")
                    or (s2cfg.mask_strategy == "corrmask" and cmcfg.provider in ("teacher_confidence", "croco_attention")))
    teacher = bb.build_teacher(cfg["teacher_checkpoint"]) if need_teacher else None
    share_enc = bb.can_share_encoder(student, teacher) if teacher is not None else (bb.name == "croco" and trainable["encoder_frozen"])
    groups = bb.param_groups(student, cfg["lr"], cfg["lr"] * cfg.get("lr_encoder_scale", 0.1), cfg["weight_decay"])
    opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "names"} for g in groups],
                            betas=tuple(cfg.get("betas", [0.9, 0.999])))
    total = cfg["steps"]
    stop = min(total, a.max_steps) if a.max_steps else total
    train_split = cfg.get("train_split", "train")   # "train" (scene-disjoint from val) or "train_all" (final model)
    frames = load_split(train_split)
    ds = TwinStereoDataset(frames, cfg["crop"], s2cfg, total, seed, cfg.get("geometry", "track3"))
    loader = DataLoader(ds, batch_size=cfg["batch_size"], sampler=StepSampler(len(frames), cfg["batch_size"], total, seed),
                        num_workers=cfg.get("workers", 6), pin_memory=True, drop_last=False, persistent_workers=False,
                        prefetch_factor=2)
    feat_blocks = L.get("feat_blocks", [-1, -2]) if L.get("lambda_feat", 0) > 0 else None
    AS = cfg.get("attn_sup") or {}
    CD = cfg.get("corrdrop") or {}     # correspondence dropout: {p, area} (needs care.variant dropout | fallback)
    if CD.get("p", 0) > 0:
        care_variant = getattr(getattr(getattr(student, "net", None), "care", None), "cfg", None)
        assert care_variant is not None and care_variant.variant in ("dropout", "fallback"), "corrdrop needs care.variant dropout|fallback"
    as_blocks = set(AS.get("blocks") or [])
    MONO = cfg.get("mono") or {}
    lam_mono = float(MONO.get("lambda_mono", 0.5)) if MONO else 0.0
    if MONO:
        print("MONO", json.dumps(dict(cfg=student.net.mono.cfg.to_dict(), n_params=sum(p.numel() for p in student.net.mono.parameters()),
                                      n_copied_from_stereo_head=student.net.mono.n_copied, lambda_mono=lam_mono)), flush=True)

    def as_caches(cache):
        # attention-supervision caches restricted to attn_sup.blocks (CARE may be installed on more blocks)
        return [dict(c) for k, c in care_block_items(cache) if k is None or not as_blocks or k in as_blocks]

    def git_hash():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "git-unavailable"

    sources = [Path(__file__), ROOT / "track2/s2aug.py", ROOT / "track2/stereo_corrmask.py", ROOT / "track2/losses.py",
               ROOT / "track2/backbones.py", ROOT / "track2/twin_data.py", ROOT / "track2/croco_model.py",
               ROOT / "track2/q4_primitives.py", ROOT / "track2/care.py", ROOT / "track2/mono.py"]
    (out / "sources").mkdir()
    for p in sources:
        shutil.copy2(p, out / "sources" / p.name)
    contract = dict(config=cfg, command=" ".join([sys.executable] + sys.argv), git=git_hash(), seed=seed,
                    host=platform.node(), gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, torch=torch.__version__,
                    visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    croco_source=json.loads((ROOT / "third_party/croco-source.json").read_text()),
                    init_checkpoint_sha256=sha256(cfg["init_checkpoint"]),
                    teacher_checkpoint_sha256=sha256(cfg["teacher_checkpoint"]) if teacher is not None else None,
                    split=dict(train=str(ROOT / f"splits/track2_{train_split}.txt"), n_train=len(frames),
                               train_sha256=sha256(ROOT / f"splits/track2_{train_split}.txt"),
                               val_sha256=sha256(ROOT / "splits/track2_val.txt"),
                               corruption_seeds_sha256=sha256(ROOT / "splits/track2_val_corruption_seeds.json")),
                    s2aug=s2cfg.to_dict(), corrmask=cmcfg.to_dict(), trainable=trainable, teacher_used=teacher is not None,
                    shared_clean_encoder=share_enc, optimizer=dict(type="AdamW", groups=[
                        dict(scope=g["scope"], lr=g["lr"], wd=g["weight_decay"], n=sum(p.numel() for p in g["params"])) for g in groups]),
                    stop_step=stop, source_sha256={p.name: sha256(p) for p in sources})
    (out / "contract.json").write_text(json.dumps(contract, indent=2, default=str))
    print("CONTRACT", json.dumps(dict(trainable=trainable, teacher=teacher is not None, share_enc=share_enc)), flush=True)

    if teacher is not None:
        assert all(not p.requires_grad for p in teacher.parameters())
    log = (out / "train.jsonl").open("a")
    step = 0
    t_start = time.time()
    first = True
    def batches():
        if a.overfit:
            b0 = next(iter(loader))
            for i in range(a.overfit):
                yield {**b0, "step": torch.full_like(b0["step"], i)}
        else:
            yield from loader
    if a.overfit:
        stop = a.overfit
    for batch in batches():
        step = int(batch["step"][0])
        if step >= stop:
            break
        tick = time.time()
        bb.train_mode(student)
        f_lr = lr_at(step, total, cfg.get("warmup_steps", 0), cfg.get("schedule", "cosine"))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f_lr
        r_cons = ramp(step, total, L.get("ramp_frac", 0.0))
        lam_corr, lam_delta = L.get("lambda_corr", 0) * r_cons, L.get("lambda_delta", 0) * r_cons
        lam_anchor, lam_feat = L.get("lambda_anchor", 0), L.get("lambda_feat", 0) * r_cons
        lam_aepi, lam_acons = AS.get("lambda_epi", 0.0) * r_cons, AS.get("lambda_cons", 0.0) * r_cons
        probe_rng = tuple(int(x) for x in a.grad_probe.split(":")) if a.grad_probe else (0, 0)
        train_params = [p for g in opt.param_groups for p in g["params"]]

        def gnorm_of(v):
            gs = torch.autograd.grad(v, train_params, retain_graph=True, allow_unused=True)
            return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
        Lc, Rc = batch["clean_left"].cuda(non_blocking=True), batch["clean_right"].cuda(non_blocking=True)
        Lx, Rx = batch["corr_left"].cuda(non_blocking=True), batch["corr_right"].cuda(non_blocking=True)
        gt = batch["disp"].cuda(non_blocking=True)
        assert Ls.valid_gt(gt).any(), "empty valid mask"
        opt.zero_grad(set_to_none=True)
        rec = dict(step=step, lr_factor=f_lr, lam_corr=lam_corr, lam_delta=lam_delta, lam_anchor=lam_anchor, lam_feat=lam_feat,
                   aug=list(batch["aug"]), severity=[float(s) for s in batch["severity"]],
                   mask_event=int(batch["mask_event"].sum()), overlay=[o for o in batch.get("overlay", []) if o != "none"],
                   erase=int(batch["erase"].sum()) if "erase" in batch else 0)

        # micro-batching (gradient accumulation) keeps the effective batch; losses are weighted by chunk share
        full = dict(Lc=Lc, Rc=Rc, Lx=Lx, Rx=Rx, gt=gt)
        B = gt.shape[0]
        mb = int(cfg.get("micro_batch") or B)
        for c0 in range(0, B, mb):
            sl = slice(c0, min(B, c0 + mb))
            frac = (sl.stop - sl.start) / B
            Lc, Rc, Lx, Rx, gt = (full[k][sl] for k in ("Lc", "Rc", "Lx", "Rx", "gt"))
            mbatch = {k: batch[k][sl] for k in ("mask_event", "mask_world", "mask_seed")}
            # 1-2: shared clean encoding and teacher
            enc_c = bb.encode(student, Lc, Rc) if share_enc else None
            T = None
            want_attn = s2cfg.mask_strategy == "corrmask" and cmcfg.provider == "croco_attention"
            if teacher is not None:
                with torch.no_grad():
                    T = bb.forward(teacher, Lc, Rc, enc=enc_c, want_attn=want_attn)
                rec["teacher_err_mean"] = float((T.disp - torch.nan_to_num(gt))[Ls.valid_gt(gt)].abs().mean())
            # 3: CorrMask / random mask placement on the corrupted pair
            if s2cfg.mask_strategy != "none" and bool(mbatch["mask_event"].any()):
                prov = "random" if s2cfg.mask_strategy == "random" else cmcfg.provider
                imp = importance(prov, Lc, gt, teacher_disp=T.disp if T is not None else None,
                                 attn_tokens=T.attn if T is not None else None, tau=cmcfg.tau)
                seeds = [int(s) for s in mbatch["mask_seed"]]
                masks, _ = sample_masks(imp, seeds, cmcfg if prov != "random" else CorrMaskConfig(provider="random"))
                Lx, Rx, _ = apply_mask_events(Lx, Rx, gt, masks, mbatch["mask_world"], mbatch["mask_event"], seeds, cmcfg,
                                              transport=s2cfg.transport, occlusion_tol=s2cfg.occlusion_tol)
                ev = mbatch["mask_event"].cuda()
                # importance inside/outside the placed boxes, measured with a fixed backbone-agnostic
                # reference (Track 3-style GT priority) and with the provider actually used
                ref = mask_stats(importance("gt_priority", Lc, gt), masks, ev)
                rec.update(mask_area=ref["mask_area"], gtprio_in=ref["imp_in"], gtprio_out=ref["imp_out"])
                if prov != "random":
                    used = mask_stats(imp, masks, ev)
                    rec.update(imp_in=used["imp_in"], imp_out=used["imp_out"])

            # 4: clean student branch
            do_probe = probe_rng[0] <= step < probe_rng[1] and c0 == 0
            with (care_recording(student.net) if AS else nullcontext({})) as cache:
                S = bb.forward(student, Lc, Rc, enc=enc_c, want_feats=feat_blocks)
                caches_c = as_caches(cache) if AS else []
            assert torch.isfinite(S.disp).all(), "non-finite clean prediction"
            gv = care_gate_values(student.net)
            if gv:
                rec.update(g_clean=float(torch.stack(gv).mean()), g_clean_last=float(gv[-1].mean()))
            l_clean = bb.native_loss(S, gt)
            loss_c = L.get("lambda_clean", 1.0) * l_clean
            if S.extra:
                l_mono_c = bb.mono_loss(S, gt)
                loss_c = loss_c + lam_mono * l_mono_c
                with torch.no_grad():
                    v = Ls.valid_gt(gt)
                    g0 = torch.nan_to_num(gt)
                    rec.update(l_mono_clean=float(l_mono_c), w_clean=float(S.extra["w"].mean()),
                               mono_err_clean=float((S.extra["d_mono"] - g0)[v].abs().mean()),
                               st_err_clean=float((S.extra["d_st"] - g0)[v].abs().mean()))
            a_clean_attn = []
            if AS:
                Ttok, vtok = token_targets(gt, caches_c[0]["grid"])
                l_aepi_c = sum(full_attention_epi_loss(c["attn_full"], Ttok, vtok) for c in caches_c) / len(caches_c)
                loss_c = loss_c + lam_aepi * 0.5 * l_aepi_c
                a_clean_attn = [c["attn_full"].detach() for c in caches_c]
                with torch.no_grad():
                    idx = torch.arange(Ttok.shape[1], device=gt.device)
                    row = lambda A: A.reshape(A.shape[0], Ttok.shape[1], Ttok.shape[2], Ttok.shape[1], Ttok.shape[2])[:, idx, :, idx, :].permute(1, 0, 2, 3)
                    mass = lambda A: float(((row(A) * (Ttok > 0)).sum(-1))[vtok].mean()) if vtok.any() else float("nan")
                    rec["attn_gt_mass_clean"] = sum(mass(c["attn_full"]) for c in caches_c) / len(caches_c)
                rec["l_attn_epi_clean"] = float(l_aepi_c)
                if do_probe:
                    rec["gradnorm_clean"] = gnorm_of(l_clean)
                    rec["gradnorm_attn_epi_clean"] = gnorm_of(l_aepi_c)
            if lam_anchor > 0:
                l_anchor, w_anchor = Ls.anchor_loss(S.disp, T.disp, gt, L["tau_anchor"], L.get("anchor_w_min", 0.0))
                loss_c = loss_c + lam_anchor * l_anchor
                v = Ls.valid_gt(gt)
                rec.update(l_anchor=float(l_anchor), w_anchor_mean=float(w_anchor[v].mean()),
                           w_anchor_q10=float(torch.quantile(w_anchor[v][::97].float(), .1)))
            assert torch.isfinite(loss_c), "non-finite clean loss"
            (frac * loss_c).backward()
            d_clean = S.disp.detach()
            z_clean = [f.detach() for f in S.feats]
            rec.update(l_clean=float(l_clean), disp_min=float(d_clean.min()), disp_max=float(d_clean.max()))
            del S, loss_c, caches_c
            grads_after_clean = {n: p.grad.detach().clone() for n, p in list(student.named_parameters())[-4:] if p.grad is not None} if first else None

            # 5: corrupted student branch (skipped for single-branch clean training, twin: false)
            if not cfg.get("twin", True):
                X = None
                rec.update(l_corr=float("nan"), pred_diff_clean_corr=float("nan"))
            enc_x = bb.encode(student, Lx, Rx) if share_enc and cfg.get("twin", True) else None
            qdrop = None
            if cfg.get("twin", True) and CD.get("p", 0) > 0:
                # correspondence dropout on the corrupted branch only (deterministic in the sample seed)
                grid = (Lx.shape[-2] // 16, Lx.shape[-1] // 16)
                qdrop = sample_query_drop([int(s) for s in mbatch["mask_seed"]], grid, CD["p"],
                                          tuple(CD.get("area", [0.05, 0.3]))).cuda()
            kmask = None
            if qdrop is not None and CD.get("keys", "epipolar") == "random":
                # control: same queries, same number of hidden keys per query, keys drawn uniformly instead of the band
                kmask = sample_random_key_mask(qdrop, [int(s) for s in mbatch["mask_seed"]], grid, care_variant.drop_band)
                if bool(qdrop.any()):
                    band = drop_mask(qdrop, grid[0], grid[1], care_variant.drop_band)
                    rec.update(drop_keys_struct=float(band.sum(-1)[qdrop].float().mean()),
                               drop_keys_actual=float(kmask.sum(-1)[qdrop].float().mean()),
                               drop_keys_in_band=float((kmask & band).sum(-1)[qdrop].float().mean()))
            with care_keydrop(student.net, qdrop, kmask), (care_recording(student.net) if AS else nullcontext({})) as cache:
                X = bb.forward(student, Lx, Rx, enc=enc_x, want_feats=feat_blocks) if cfg.get("twin", True) else None
                caches_x = as_caches(cache) if AS else []
            loss_x = 0
            if X is not None:
                assert torch.isfinite(X.disp).all(), "non-finite corrupted prediction"
                l_corr = bb.native_loss(X, gt)
                loss_x = loss_x + lam_corr * l_corr
                rec["l_corr"] = float(l_corr)
                if X.extra:
                    l_mono_x = bb.mono_loss(X, gt)
                    loss_x = loss_x + lam_mono * l_mono_x
                    with torch.no_grad():
                        v = Ls.valid_gt(gt)
                        g0 = torch.nan_to_num(gt)
                        rec.update(l_mono_corr=float(l_mono_x), w_corr=float(X.extra["w"].mean()),
                                   mono_err_corr=float((X.extra["d_mono"] - g0)[v].abs().mean()),
                                   st_err_corr=float((X.extra["d_st"] - g0)[v].abs().mean()),
                                   mono_diff=float((X.extra["d_mono"] - d_clean).abs().mean()),
                                   st_diff=float((X.extra["d_st"] - d_clean).abs().mean()))
                gv = care_gate_values(student.net)
                if gv:
                    gm = torch.stack(gv).mean(0)                                                   # B x N over blocks
                    rec.update(g_corr=float(gm.mean()), g_corr_last=float(gv[-1].mean()))
                    if qdrop is not None and bool(qdrop.any()):
                        rec.update(g_corr_drop=float(gm[qdrop].mean()), g_corr_keep=float(gm[~qdrop].mean()),
                                   g_corr_drop_last=float(gv[-1][qdrop].mean()))
                if qdrop is not None:
                    rec.update(drop_frac=float(qdrop.any(1).float().mean()), drop_area=float(qdrop.float().mean()))
            if X is not None and AS:
                # queries whose epipolar key band is masked by correspondence dropout cannot satisfy the attention targets
                vtok_x = vtok & ~qdrop.view_as(vtok) if qdrop is not None else vtok
                l_aepi_x = sum(full_attention_epi_loss(c["attn_full"], Ttok, vtok_x) for c in caches_x) / len(caches_x)
                l_acons = sum(full_attention_consistency(ac, c["attn_full"], vtok_x) for ac, c in zip(a_clean_attn, caches_x)) / len(caches_x)
                loss_x = loss_x + lam_aepi * 0.5 * l_aepi_x + lam_acons * l_acons
                with torch.no_grad():
                    rec["attn_gt_mass_corr"] = sum(mass(c["attn_full"]) for c in caches_x) / len(caches_x)  # incl. dropped queries (diagnostic)
                rec.update(l_attn_epi_corr=float(l_aepi_x), l_attn_cons=float(l_acons))
                if do_probe:
                    rec["gradnorm_corr"] = gnorm_of(l_corr)
                    rec["gradnorm_attn_epi_corr"] = gnorm_of(l_aepi_x)
                    rec["gradnorm_attn_cons"] = gnorm_of(l_acons)
            if X is not None and L.get("lambda_delta", 0) > 0:
                l_delta, w_delta = Ls.delta_loss(X.disp, d_clean, gt, L.get("delta_weight", "valid"),
                                                 T.disp if T is not None else None, L.get("tau_delta", 1.0))
                loss_x = loss_x + lam_delta * l_delta
                rec.update(l_delta=float(l_delta), w_delta_mean=float(w_delta[Ls.valid_gt(gt)].mean()))
                if qdrop is not None and bool(qdrop.any()):
                    # fallback quality: |d_corr - d_clean| under dropped tokens vs elsewhere
                    Ht, Wt = Lx.shape[-2] // 16, Lx.shape[-1] // 16
                    pm = qdrop.view(-1, 1, Ht, Wt).repeat_interleave(16, 2).repeat_interleave(16, 3)
                    diff = (X.disp.detach() - d_clean).abs()
                    rec.update(diff_drop=float(diff[pm].mean()), diff_keep=float(diff[~pm].mean()))
                    if X.extra:
                        rec["w_corr_drop"] = float(X.extra["w"].detach()[pm].mean())
            if X is not None and lam_feat > 0:
                v = Ls.valid_gt(gt).float()
                l_feat = Ls.feature_loss(X.feats, z_clean, v, patch=bb.feat_patch)
                loss_x = loss_x + lam_feat * l_feat
                rec["l_feat"] = float(l_feat)
            if X is not None:
                rec["pred_diff_clean_corr"] = float((X.disp.detach() - d_clean).abs().mean())
            if torch.is_tensor(loss_x):
                assert torch.isfinite(loss_x), "non-finite corrupted loss"
                if first:
                    # gradient-path audit: delta must not reach d_clean, anchor must not reach teacher
                    if L.get("lambda_delta", 0) > 0:
                        assert not d_clean.requires_grad
                    if teacher is not None:
                        assert all(p.grad is None for p in teacher.parameters())
                (frac * loss_x).backward()
            del X
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], cfg.get("grad_clip") or 1e9)
        assert torch.isfinite(gn), "non-finite grad norm"
        if first:
            enc_grad = [n for n, p in student.named_parameters() if p.grad is not None and not p.requires_grad]
            assert not enc_grad
            frozen_with_grad = [n for n, p in student.named_parameters() if not p.requires_grad and p.grad is not None]
            probe = [(n, p.detach().clone()) for n, p in student.named_parameters() if p.requires_grad and p.grad is not None][:1]
            audit = dict(grad_norm=float(gn), frozen_params_with_grad=frozen_with_grad,
                         n_params_with_grad=sum(1 for p in student.parameters() if p.grad is not None),
                         teacher_grads=None if teacher is None else sum(1 for p in teacher.parameters() if p.grad is not None),
                         corrupted_branch_changed_grads=None if grads_after_clean is None else any(
                             not torch.equal(grads_after_clean[n], dict(student.named_parameters())[n].grad) for n in grads_after_clean))
        opt.step()
        if first:
            n, before = probe[0]
            audit["probe_param"] = n
            audit["probe_change"] = float((dict(student.named_parameters())[n].detach() - before).abs().max())
            assert audit["probe_change"] > 0
            (out / "gradient_audit.json").write_text(json.dumps(audit, indent=2))
            print("GRADIENT_AUDIT", json.dumps(audit), flush=True)
            first = False
        rec.update(grad_norm=float(gn), seconds=time.time() - tick, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        log.write(json.dumps(rec) + "\n")
        if step % 10 == 0 or step < 3:
            log.flush()
            eta = (time.time() - t_start) / (step + 1) * (stop - step - 1)
            extra = " ".join(f"{k}={rec[k]:.4f}" for k in ("l_attn_epi_clean", "l_attn_cons", "attn_gt_mass_clean", "attn_gt_mass_corr",
                                                           "g_clean", "g_corr", "g_corr_drop", "diff_drop", "diff_keep",
                                                           "w_clean", "w_corr", "w_corr_drop", "mono_err_clean", "mono_err_corr", "st_err_corr") if k in rec)
            print(f"step {step}/{stop} {extra} clean={rec['l_clean']:.3f} corr={rec['l_corr']:.3f} "
                  f"delta={rec.get('l_delta', 0):.3f} anchor={rec.get('l_anchor', 0):.3f} diff={rec['pred_diff_clean_corr']:.3f} "
                  f"gn={rec['grad_norm']:.2f} {rec['seconds']:.2f}s peak={rec['peak_gib']:.1f}G eta={eta/3600:.2f}h", flush=True)
        if not a.no_save and cfg.get("save_every") and (step + 1) % cfg["save_every"] == 0 and step + 1 < stop:
            bb.save(student, out / "latest.pt", dict(step=step + 1, run=str(out)))
    log.close()
    final_step = stop
    if not a.no_save:
        bb.save(student, out / "final.pt", dict(step=final_step, run=str(out), config=cfg))
        if (out / "latest.pt").exists():
            (out / "latest.pt").unlink()
    (out / "status.json").write_text(json.dumps(dict(phase="complete", steps=final_step, wall_seconds=time.time() - t_start)))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Same-checkpoint monocular ablation (Table 9): fused prediction of a full-model checkpoint vs stereo-only inference
of the SAME checkpoint with the monocular module disabled, on the 28 primary frames x (clean + 20 corruptions).

  CUDA_VISIBLE_DEVICES=<gpu> python3 scripts/three_seed/mono_same_checkpoint.py run --seed 2026      (resumable)
  python3 scripts/three_seed/mono_same_checkpoint.py summarize

For every frame-condition two inference passes of the unchanged checkpoint:
  A  scripts/final_analysis.tiled_debug -> fused prediction (asserted equal to the production tiled_predict),
                                           mono-only prediction, fusion weight w_f
  B  model.tiled_predict under track2.mono.mono_disabled -> production stereo-only output (its own tile confidences)
Metrics follow scripts/eval_suite.py (track2.val_suite.frame_metrics; delta = mean |d_cond - d_clean| over all pixels);
regions as in scripts/final_analysis.regions. Output folder: outputs/track2/three_seed/mono ($ROCO_MONO_OUT overrides)."""
import argparse
import csv
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

import common as K

OUT = Path(os.environ.get("ROCO_MONO_OUT", K.OUT / "mono"))
CONDS = ("clean",) + K.CORRUPTIONS20
HB, HMIN = 20000, 0.90        # pooled histogram of w_f on [0.90, 1.00]; anything below 0.90 is counted separately


def run(seed):
    import torch
    sys.path.insert(0, str(K.ROOT / "scripts"))
    import final_analysis as FA
    from track2 import val_suite as vs
    from track2.croco_model import build_croco_stereo
    from track2.mono import mono_disabled
    from track2.spring_data import read_frame

    def q(t, p):
        t = t.flatten()
        if t.numel() > 4_000_000:
            t = t[:: t.numel() // 2_000_000]
        return float(torch.quantile(t.float(), p))

    ckpt = K.run("full", seed)["train"] / "final.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT / f"same_checkpoint_rows_seed{seed}.jsonl"
    hist_path = OUT / f"fusion_weight_hist_seed{seed}.npz"
    done = set()
    if rows_path.exists():
        done = {(r["frame"], r["condition"]) for r in map(json.loads, rows_path.read_text().splitlines())}
    sha_before = K.sha256(ckpt)
    contract = dict(seed=seed, checkpoint_sha256=sha_before, conditions=list(CONDS), robust20_manifest_sha256=K.sha256(K.R20 / "manifest.json"),
                    device=torch.cuda.get_device_name(0), inference="tiles 352x704, overlap 0.7, tile_batch 1, fp32")
    (OUT / f"same_checkpoint_contract_seed{seed}.json").write_text(json.dumps(contract, indent=1))
    model, _ = build_croco_stereo(str(ckpt), "cuda")
    model.eval()
    net = model.net
    state0 = {k: v.clone() for k, v in list(net.state_dict().items())[:8]}
    to = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda()
    hist = {(c, r): torch.zeros(HB, dtype=torch.float64) for c in CONDS for r in ("all", "sky", "textureless")}
    below = {k: 0 for k in hist}
    if hist_path.exists():
        z = np.load(hist_path, allow_pickle=True)
        for k in hist:
            hist[k] = torch.from_numpy(z[f"{k[0]}|{k[1]}"]).double()
            below[k] = int(z[f"{k[0]}|{k[1]}|below"])
    frames = FA.frames()
    assert len(frames) == K.N_PRIMARY
    t0 = time.time()
    for i, fr in enumerate(frames):
        need = [c for c in CONDS if (fr.key, c) not in done]
        if not need:
            continue
        item = read_frame(fr, full_gt=True)
        gfull, gstr = torch.from_numpy(item["disp_full"]).cuda(), torch.from_numpy(item["disp"]).cuda()
        gt = item["disp"]
        clean = {}
        new = []
        for cond in CONDS:                      # clean first: its two predictions are the references of the deltas
            if cond != "clean" and cond not in need:
                continue
            L, R = FA.load_pair(fr, cond, item)
            with torch.no_grad():
                dbg = {k: v.float() for k, v in FA.tiled_debug(model, to(L), to(R)).items()}
                with mono_disabled(net):
                    stp = model.tiled_predict(to(L), to(R), overlap=0.7, tile_batch=1)[0, 0].float()
                assert net.mono.enabled, "mono_disabled must be reversible"
                check = None
                if i == 0:                      # the debug loop must equal the production fused path
                    ref = model.tiled_predict(to(L), to(R), overlap=0.7, tile_batch=1)[0, 0].float()
                    check = float((ref - dbg["pred"]).abs().max())
                    assert check < 1e-3, check
            fused, mono, w = dbg["pred"], dbg["d_mono"], dbg["w"]
            assert all(torch.isfinite(x).all() for x in (fused, stp, mono, w))
            if cond == "clean":
                clean = dict(fused=fused, st=stp)
                if "clean" not in need:
                    continue
            reg = {k: torch.from_numpy(v).cuda() for k, v in FA.regions(gt, L).items()}
            valid = reg["valid"]
            gtt = torch.nan_to_num(gstr)
            d = (fused - stp).abs()
            r = dict(seed=seed, frame=fr.key, condition=cond, debug_vs_production_fused_max=check)
            r.update({f"fused_{k}": v for k, v in vs.frame_metrics(fused, gfull, gstr).items() if k in ("abs", "d1", "px1")})
            r.update({f"stereo_{k}": v for k, v in vs.frame_metrics(stp, gfull, gstr).items() if k in ("abs", "d1", "px1")})
            if cond != "clean":
                r.update(fused_delta=float((fused - clean["fused"]).abs().mean()), stereo_delta=float((stp - clean["st"]).abs().mean()))
            r.update(diff_mean=float(d.mean()), diff_max=float(d.max()), diff_p99=q(d, 0.99),
                     debug_stereo_vs_production_stereo_mean=float((dbg["d_st"] - stp).abs().mean()),
                     debug_stereo_vs_production_stereo_max=float((dbg["d_st"] - stp).abs().max()))
            e_f, e_s, e_m = (fused - gtt).abs(), (stp - gtt).abs(), (mono - gtt).abs()
            for rn in ("valid", "sky", "nonsky", "textureless", "textured"):
                m = reg[rn] & valid
                n = int(m.sum())
                r[f"n_{rn}"] = n
                for tag, e in (("fused", e_f), ("stereo", e_s), ("mono", e_m)):
                    r[f"err_{tag}_{rn}"] = float(e[m].mean()) if n else None
                r[f"diff_mean_{rn}"] = float(d[m].mean()) if n else None
                r[f"diff_max_{rn}"] = float(d[m].max()) if n else None
            ch = (e_f - e_s)[valid]                 # < 0: fusion reduces the absolute error against GT
            r.update(frac_fused_better=float((ch < 0).float().mean()), frac_fused_worse=float((ch > 0).float().mean()),
                     frac_better_gt_0p01=float((ch < -0.01).float().mean()), frac_worse_gt_0p01=float((ch > 0.01).float().mean()),
                     frac_better_gt_0p1=float((ch < -0.1).float().mean()), frac_worse_gt_0p1=float((ch > 0.1).float().mean()),
                     mean_err_change=float(ch.mean()), frac_mono_better_than_stereo=float((e_m < e_s)[valid].float().mean()))
            for rn in ("all", "sky", "textureless"):
                m = reg[rn]
                if not bool(m.any()):
                    continue
                wm = w[m]
                r.update({f"w_mean_{rn}": float(wm.mean()), f"w_min_{rn}": float(wm.min()), f"w_p05_{rn}": q(wm, 0.05),
                          f"w_p50_{rn}": q(wm, 0.5), f"w_p95_{rn}": q(wm, 0.95), f"w_lt_0p99_{rn}": float((wm < .99).float().mean()),
                          f"w_lt_0p95_{rn}": float((wm < .95).float().mean()), f"w_lt_0p90_{rn}": float((wm < .90).float().mean()),
                          f"w_n_{rn}": int(m.sum())})
                hist[(cond, rn)] += torch.histc(wm.double().clamp(HMIN, 1.0), bins=HB, min=HMIN, max=1.0).cpu()
                below[(cond, rn)] += int((wm < HMIN).sum())
            new.append(r)
        with rows_path.open("a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
        np.savez(hist_path, **{f"{k[0]}|{k[1]}": v.numpy() for k, v in hist.items()}, **{f"{k[0]}|{k[1]}|below": np.int64(v) for k, v in below.items()})
        print(f"[seed {seed}] {i + 1}/{len(frames)} {fr.key} {len(new)} rows {time.time() - t0:.0f}s", flush=True)
    # the checkpoint and the loaded weights must be untouched
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in state0.items())
    assert K.sha256(ckpt) == sha_before
    (OUT / f"same_checkpoint_DONE_seed{seed}").write_text(time.strftime("%F %T"))
    print("DONE", seed)


def pooled_quantiles(hist, below, qs):
    tot = hist.sum() + below
    if tot == 0:
        return [float("nan")] * len(qs)
    cum = np.cumsum(hist) + below
    edges = np.linspace(HMIN, 1.0, len(hist) + 1)
    return [float(edges[min(len(hist) - 1, int(np.searchsorted(cum, q * tot)))] + (edges[1] - edges[0]) / 2) for q in qs]


def summarize():
    L = ["# Same-checkpoint monocular ablation", "",
         "Production `tiled_predict` (352×704 tiles, overlap 0.7, fp32) of each unchanged full-model checkpoint, once as is (fused) and once under "
         "`track2.mono.mono_disabled` (stereo-only: the stereo head output and its own confidences drive the tile blending). 28 primary frames × "
         "(clean + 20 corruptions); local development results.", ""]
    comp = [["seed", "condition", "n_frames", "fused_abs", "stereo_abs", "fused_d1", "stereo_d1", "fused_delta", "stereo_delta", "mean_abs_diff_px", "max_abs_diff_px",
             "p99_abs_diff_px_mean", "diff_mean_sky", "diff_max_sky", "diff_mean_textureless", "diff_max_textureless", "frac_fused_better", "frac_fused_worse",
             "frac_better_gt_0p01px", "frac_worse_gt_0p01px", "frac_better_gt_0p1px", "frac_worse_gt_0p1px", "mean_err_change_px",
             "w_mean", "w_min", "w_p05", "w_median", "w_p95", "w_frac_lt_0p99", "w_frac_lt_0p95", "w_frac_lt_0p90", "w_mean_sky", "w_min_sky", "w_mean_textureless", "w_min_textureless",
             "err_fused_valid", "err_stereo_valid", "err_mono_valid", "err_fused_sky", "err_stereo_sky", "err_mono_sky",
             "err_fused_textureless", "err_stereo_textureless", "err_mono_textureless", "err_fused_textured", "err_stereo_textured", "err_mono_textured",
             "debug_stereo_vs_production_stereo_mean", "debug_stereo_vs_production_stereo_max"]]
    summ = {}
    for seed in K.SEEDS:
        p = OUT / f"same_checkpoint_rows_seed{seed}.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        if not (OUT / f"same_checkpoint_DONE_seed{seed}").exists() or len(rows) != K.N_PRIMARY * 21:
            L.append(f"- seed {seed}: incomplete ({len(rows)}/588 rows), not summarised")
            continue
        hz = np.load(OUT / f"fusion_weight_hist_seed{seed}.npz", allow_pickle=True)
        per = {}
        for r in rows:
            per.setdefault(r["condition"], []).append(r)
        assert all(len(v) == K.N_PRIMARY for v in per.values()) and len(per) == 21
        g = lambda c, k: st.fmean(x[k] for x in per[c] if x.get(k) is not None) if any(x.get(k) is not None for x in per[c]) else float("nan")
        wavg = lambda c, k, nk: (sum(x[k] * x[nk] for x in per[c] if x.get(k) is not None) / max(1, sum(x[nk] for x in per[c] if x.get(k) is not None)))
        # the fused path must agree with the regular evaluation of the same checkpoint
        ev = K.run("full", seed)["eval"] / "rows.jsonl"
        chk = None
        if ev.exists():
            old = {(r["frame"], r["condition"]): r for r in map(json.loads, ev.read_text().splitlines()) if r["frame"].split(":")[0] in K.PRIMARY}
            chk = max(max(abs(r["fused_abs"] - old[(r["frame"], r["condition"])]["abs"]),
                          abs(r.get("fused_delta", 0.0) - old[(r["frame"], r["condition"])].get("delta", 0.0))) for r in rows)
        fogq = pooled_quantiles(hz["fog|all"], int(hz["fog|all|below"]), (0.5,))[0]
        summ[seed] = dict(clean_abs_fused=g("clean", "fused_abs"), clean_abs_stereo=g("clean", "stereo_abs"), clean_d1_fused=g("clean", "fused_d1"), clean_d1_stereo=g("clean", "stereo_d1"),
                          dabs20_fused=st.fmean(g(c, "fused_delta") for c in K.CORRUPTIONS20), dabs20_stereo=st.fmean(g(c, "stereo_delta") for c in K.CORRUPTIONS20),
                          max_dev_from_evaluation=chk, diff_mean_all=st.fmean(r["diff_mean"] for r in rows), diff_max_all=max(r["diff_max"] for r in rows),
                          w_min=min(r["w_min_all"] for r in rows), w_lt_099=st.fmean(r["w_lt_0p99_all"] for r in rows), w_lt_095=st.fmean(r["w_lt_0p95_all"] for r in rows),
                          w_lt_090=st.fmean(r["w_lt_0p90_all"] for r in rows), w_median_fog=fogq,
                          fog={k: g("fog", k) for k in ("err_mono_sky", "err_stereo_sky", "err_mono_textureless", "err_stereo_textureless")},
                          per={c: {k: g(c, k) for k in ("fused_delta", "stereo_delta")} for c in per})
        for c in CONDS:
            qs = pooled_quantiles(hz[f"{c}|all"], int(hz[f"{c}|all|below"]), (0.05, 0.5, 0.95))
            comp.append([seed, c, len(per[c])] + [f"{g(c, k):.6f}" for k in ("fused_abs", "stereo_abs", "fused_d1", "stereo_d1")]
                        + [f"{g(c, 'fused_delta'):.6f}" if c != "clean" else "", f"{g(c, 'stereo_delta'):.6f}" if c != "clean" else ""]
                        + [f"{g(c, 'diff_mean'):.6f}", f"{max(x['diff_max'] for x in per[c]):.5f}", f"{g(c, 'diff_p99'):.6f}",
                           f"{g(c, 'diff_mean_sky'):.6f}", f"{max((x['diff_max_sky'] or 0) for x in per[c]):.5f}",
                           f"{g(c, 'diff_mean_textureless'):.6f}", f"{max((x['diff_max_textureless'] or 0) for x in per[c]):.5f}"]
                        + [f"{g(c, k):.6f}" for k in ("frac_fused_better", "frac_fused_worse", "frac_better_gt_0p01", "frac_worse_gt_0p01", "frac_better_gt_0p1", "frac_worse_gt_0p1", "mean_err_change")]
                        + [f"{g(c, 'w_mean_all'):.6f}", f"{min(x['w_min_all'] for x in per[c]):.6f}", f"{qs[0]:.6f}", f"{qs[1]:.6f}", f"{qs[2]:.6f}",
                           f"{g(c, 'w_lt_0p99_all'):.6g}", f"{g(c, 'w_lt_0p95_all'):.6g}", f"{g(c, 'w_lt_0p90_all'):.6g}",
                           f"{wavg(c, 'w_mean_sky', 'w_n_sky'):.6f}", f"{min((x['w_min_sky'] for x in per[c] if x.get('w_min_sky') is not None), default=float('nan')):.6f}",
                           f"{wavg(c, 'w_mean_textureless', 'w_n_textureless'):.6f}", f"{min((x['w_min_textureless'] for x in per[c] if x.get('w_min_textureless') is not None), default=float('nan')):.6f}"]
                        + [f"{g(c, f'err_{t}_{rn}'):.5f}" for rn in ("valid", "sky", "textureless", "textured") for t in ("fused", "stereo", "mono")]
                        + [f"{g(c, 'debug_stereo_vs_production_stereo_mean'):.3e}", f"{max(x['debug_stereo_vs_production_stereo_max'] for x in per[c]):.5f}"])
    if summ:
        L += ["| full-model checkpoint (seed) | Clean Abs fused / stereo-only | Clean D1 fused / stereo-only | ΔAbs20 fused / stereo-only | ΔAbs20 difference (fused − stereo-only) | mean / max pixelwise |fused − stereo-only| (px) | fused path vs regular evaluation (max |dev|) |",
              "|---|---|---|---|---|---|---|"]
        for s, v in sorted(summ.items()):
            dev = f"{v['max_dev_from_evaluation']:.2e}" if v["max_dev_from_evaluation"] is not None else "–"
            L.append(f"| {s} | {v['clean_abs_fused']:.4f} / {v['clean_abs_stereo']:.4f} | {v['clean_d1_fused']:.4f} / {v['clean_d1_stereo']:.4f} | {v['dabs20_fused']:.4f} / {v['dabs20_stereo']:.4f} | "
                     f"{v['dabs20_fused'] - v['dabs20_stereo']:+.5f} | {v['diff_mean_all']:.5f} / {v['diff_max_all']:.3f} | {dev} |")
        L += ["", "| seed | fog ΔAbs fused / stereo-only | frost | snow |", "|---|---|---|---|"]
        for s, v in sorted(summ.items()):
            L.append(f"| {s} | " + " | ".join(f"{v['per'][c]['fused_delta']:.4f} / {v['per'][c]['stereo_delta']:.4f}" for c in ("fog", "frost", "snow")) + " |")
        L += ["", "## Fusion gate (stereo weight w_f; d = w_f·d_stereo + (1 − w_f)·d_mono), all 21 conditions pooled", "",
              "| seed | min w_f | fraction of pixels with w_f < 0.99 | < 0.95 | < 0.90 | median w_f under fog |", "|---|---|---|---|---|---|"]
        for s, v in sorted(summ.items()):
            L.append(f"| {s} | {v['w_min']:.5f} | {v['w_lt_099']:.3g} | {v['w_lt_095']:.3g} | {v['w_lt_090']:.3g} | {v['w_median_fog']:.4f} |")
        L += ["", "## Error against ground truth under fog (px): monocular-only vs stereo-only prediction", "",
              "| seed | sky: mono / stereo | textureless: mono / stereo |", "|---|---|---|"]
        for s, v in sorted(summ.items()):
            f = v["fog"]
            L.append(f"| {s} | {f['err_mono_sky']:.1f} / {f['err_stereo_sky']:.1f} | {f['err_mono_textureless']:.1f} / {f['err_stereo_textureless']:.1f} |")
        L += ["", "Per-condition gate statistics, regional errors of the fused / stereo-only / monocular-only predictions and the fractions of pixels whose error "
              "improves or worsens: `mono_same_checkpoint.csv` (one row per checkpoint and condition).", "",
              "LaTeX rows of Table 9:", "", "```latex"]
        for s, v in sorted(summ.items()):
            lt = f"{100 * v['w_lt_099']:.2g}\\%" if v["w_lt_099"] >= 1e-5 else "$<$0.001\\%"
            L.append(f"{s} & {v['clean_abs_fused']:.4f} / {v['clean_abs_stereo']:.4f} & {v['dabs20_fused']:.4f} / {v['dabs20_stereo']:.4f} & {v['w_min']:.3f} & {lt} \\\\")
        L += ["```"]
    (OUT / "mono_same_checkpoint.md").write_text("\n".join(L) + "\n")
    with open(OUT / "mono_same_checkpoint.csv", "w", newline="") as f:
        csv.writer(f).writerows(comp)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--seed", type=int, choices=K.SEEDS)
    a = ap.parse_args()
    if a.cmd == "run":
        assert a.seed, "--seed is required for run"
        run(a.seed)
    else:
        summarize()

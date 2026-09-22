#!/usr/bin/env python3
"""Post-hoc paper analyses for the FIXED final model (no training, no model change, local validation data only).

  predict : E0 and Full on the 28 primary validation frames x {clean, fog, frost, snow} (frozen robust20 cache);
            Full additionally exposes the tile-fused stereo / mono predictions and the fusion weight w_f by reading the
            existing debug cache net.mono.last after every tile forward (same tiling and confidence weights as
            Track2CroCoStereo.tiled_predict; the fused output is asserted to equal tiled_predict). Cached as float16.
  stats   : regional breakdown, fusion-weight statistics, stereo/mono/fused error vs GT, ranking of qualitative candidates.
  figures : qualitative panels, fusion maps, publication figure candidates.
  latency : per-tile and full-frame latency of stereo-only vs full model.

Definitions are the ones already used in scripts/analyze_evidence_free.py: sky = finite GT disparity == 0,
textureless = 15x15 grey local std < 2 on the (corrupted) LEFT input of that condition. GT = Spring disp1_left strided
to the image grid (track2.spring_data.read_frame), NaN = invalid.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__import__("os").environ.get("ROCO_TRACK2_WORK", ROOT))   # checkpoints/, outputs/, third_party/croco live here
sys.path.insert(0, str(ROOT))
from track2.croco_model import build_croco_stereo, normalize_rgb255  # noqa: E402
from track2.mono import mono_disabled  # noqa: E402
from track2.spring_data import load_split, read_frame  # noqa: E402
from stereoflow.engine import _overlapping, _crop  # noqa: E402

OUT = WORK / "outputs/track2/final_analysis"
R20 = WORK / "outputs/track2/robust20_val"
PRIMARY = ("0038", "0039", "0045", "0047")
CONDS = ("clean", "fog", "frost", "snow")
MODELS = {"E0": WORK / "checkpoints/track2/croco_clean_teacher.pt",
          "Full": WORK / "outputs/track2/track2_croco_x7_x6b_mono/stageC_seed2026/final.pt"}
THR, K = 2.0, 15


def frames():
    return [f for f in load_split("val") if f.scene in PRIMARY]


def load_pair(fr, cond, item):
    if cond == "clean":
        return item["left"], item["right"]
    g = lambda s: np.asarray(Image.open(R20 / cond / fr.scene / f"frame_{s}_{fr.frame:04d}.png").convert("RGB")).copy()
    return g("left"), g("right")


def local_std(img255: np.ndarray) -> np.ndarray:
    g = torch.from_numpy(img255).float().permute(2, 0, 1)[None]
    g = 0.299 * g[:, :1] + 0.587 * g[:, 1:2] + 0.114 * g[:, 2:3]
    m = F.avg_pool2d(F.pad(g, (K // 2,) * 4, mode="reflect"), K, 1)
    m2 = F.avg_pool2d(F.pad(g * g, (K // 2,) * 4, mode="reflect"), K, 1)
    return (m2 - m * m).clamp_min(0).sqrt()[0, 0].numpy()


@torch.no_grad()
def tiled_debug(model, L255, R255, overlap=0.7):
    """Same loop as Track2CroCoStereo.tiled_predict (tile_batch 1); also fuses d_st, d_mono, w_f with the same weights."""
    beta, betasig = map(float, model.meta["tile_conf_mode"][len("conf_expsigmoid_"):].split("_"))
    a1, a2 = normalize_rgb255(L255.float()), normalize_rgb255(R255.float())
    B, _, H, W = a1.shape
    wh, ww = model.crop
    mono = getattr(model.net, "mono", None)
    names = ["pred"] + (["d_st", "d_mono", "w"] if mono is not None else [])
    acc = {n: a1.new_zeros((H, W)) for n in names}
    wsum = a1.new_zeros((H, W)) + 1e-16
    for sy in _overlapping(H, wh, overlap):
        for sx in _overlapping(W, ww, overlap):
            out = model.net(_crop(a1, sy, sx), _crop(a2, sy, sx)).float()
            conf = torch.exp(-beta * 2 * (torch.sigmoid(out[0, 1] / betasig) - 0.5))
            acc["pred"][sy, sx] += out[0, 0] * conf
            if mono is not None:
                acc["d_st"][sy, sx] += mono.last["d_st"][0, 0].float() * conf
                acc["d_mono"][sy, sx] += mono.last["d_mono"][0, 0].float() * conf
                acc["w"][sy, sx] += mono.last["w"][0, 0].float() * conf
            wsum[sy, sx] += conf
    return {n: (v / wsum) for n, v in acc.items()}


def cmd_predict(a):
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    to = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda()
    for name, ck in MODELS.items():
        model, _ = build_croco_stereo(str(ck), "cuda")
        model.eval()
        for i, fr in enumerate(frames()):
            item = read_frame(fr)
            for cond in CONDS:
                p = cache / f"{name}_{fr.scene}_{fr.frame:04d}_{cond}.npz"
                if p.exists():
                    continue
                L, R = load_pair(fr, cond, item)
                out = tiled_debug(model, to(L), to(R))
                if i == 0:  # the debug loop must reproduce the production inference path exactly
                    ref = model.tiled_predict(to(L), to(R), overlap=0.7, tile_batch=1)[0, 0]
                    err = float((ref - out["pred"]).abs().max())
                    assert err < 1e-3, err
                    print(f"[{name}] {cond}: max |tiled_debug - tiled_predict| = {err:.2e}", flush=True)
                np.savez_compressed(p, **{k: v.cpu().numpy().astype(np.float32 if k == "w" else np.float16) for k, v in out.items()})
            print(f"[{name}] {i + 1}/28 {fr.key}", flush=True)
        del model
        torch.cuda.empty_cache()
    print("PREDICT DONE")


def load_pred(name, fr, cond):
    z = np.load(OUT / "cache" / f"{name}_{fr.scene}_{fr.frame:04d}_{cond}.npz")
    return {k: z[k].astype(np.float32) for k in z.files}


def regions(gt, left255):
    fin = np.isfinite(gt)
    sky = fin & (gt == 0)
    free = local_std(left255) < THR
    return dict(all=np.ones_like(fin), valid=fin, sky=sky, nonsky=fin & ~sky, textureless=free, textured=~free)


def cmd_stats(a):
    rows, wrows, erows, cand = [], [], [], []
    wvals = {c: [] for c in CONDS}
    for fr in frames():
        item = read_frame(fr)
        gt = item["disp"]
        pred = {(m, c): load_pred(m, fr, c) for m in MODELS for c in CONDS}
        for cond in CONDS:
            L, _ = load_pair(fr, cond, item)
            reg = regions(gt, L)
            for m in MODELS:
                pc, px = pred[(m, "clean")]["pred"], pred[(m, cond)]["pred"]
                diff = np.abs(px - pc)
                tot_all, tot_valid = float(diff.sum()), float(diff[reg["valid"]].sum())
                if cond != "clean":
                    for rn, mask in reg.items():
                        n = int(mask.sum())
                        rows.append(dict(model=m, frame=fr.key, condition=cond, region=rn, pixel_frac=n / mask.size,
                                         mean_abs_change=float(diff[mask].mean()) if n else float("nan"),
                                         share_of_total_all_pixels=float(diff[mask].sum()) / tot_all if tot_all > 0 else float("nan"),
                                         share_of_total_valid_pixels=float(diff[mask & reg["valid"]].sum()) / tot_valid if tot_valid > 0 else float("nan"),
                                         mean_clean_disp=float(pc[mask].mean()) if n else float("nan"),
                                         mean_corr_disp=float(px[mask].mean()) if n else float("nan")))
                # C2: error vs GT of the fused / stereo-only / mono-only predictions
                comps = {"fused": px} if m == "E0" else {"fused": px, "stereo_only": pred[(m, cond)]["d_st"], "mono_only": pred[(m, cond)]["d_mono"]}
                for cn, d in comps.items():
                    for rn in ("valid", "sky", "nonsky", "textureless", "textured"):
                        mask = reg[rn] & reg["valid"]
                        erows.append(dict(model=m, frame=fr.key, condition=cond, prediction=cn, region=rn,
                                          abs_err_vs_gt=float(np.abs(d - gt)[mask].mean()) if mask.any() else float("nan"), n=int(mask.sum())))
            w = pred[("Full", cond)]["w"]
            dsm = np.abs(pred[("Full", cond)]["d_st"] - pred[("Full", cond)]["d_mono"])
            wvals[cond].append(w[::4, ::4].ravel())
            for rn, mask in reg.items():
                if mask.any():
                    wrows.append(dict(frame=fr.key, condition=cond, region=rn, pixel_frac=float(mask.mean()), mean_w=float(w[mask].mean()),
                                      min_w=float(w[mask].min()), frac_w_lt_0p99=float((w[mask] < .99).mean()), frac_w_lt_0p95=float((w[mask] < .95).mean()),
                                      frac_w_lt_0p90=float((w[mask] < .90).mean()), mean_abs_st_minus_mono=float(dsm[mask].mean())))
            if cond != "clean":
                e0 = float(np.abs(pred[("E0", cond)]["pred"] - pred[("E0", "clean")]["pred"]).mean())
                fu = float(np.abs(pred[("Full", cond)]["pred"] - pred[("Full", "clean")]["pred"]).mean())
                cand.append(dict(frame=fr.key, condition=cond, e0_delta=e0, full_delta=fu, reduction=e0 - fu, reduction_ratio=(e0 - fu) / e0 if e0 > 0 else 0.0,
                                 sky_frac=float(reg["sky"].mean()), textureless_frac=float(reg["textureless"].mean())))
        print("stats", fr.key, flush=True)

    def dump(name, rs):
        with (OUT / name).open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rs[0]))
            wr.writeheader()
            wr.writerows(rs)
    dump("regional_stats.csv", rows)
    dump("fusion_weights.csv", wrows)
    dump("mono_stereo_fused_error.csv", erows)
    # ---- candidate ranking: per corruption, frames with sky >= 5 %, top-5 by E0 change, pick the median reduction ratio
    sel = {}
    for c in CONDS[1:]:
        cs = sorted([r for r in cand if r["condition"] == c], key=lambda r: -r["e0_delta"])
        for i, r in enumerate(cs):
            r["rank_by_e0_delta"] = i + 1
        top = [r for r in cs if r["sky_frac"] >= 0.05][:5]
        pick = sorted(top, key=lambda r: r["reduction_ratio"])[len(top) // 2]
        pick["selected"] = 1
        sel[c] = pick["frame"]
    for r in cand:
        r.setdefault("selected", 0)
        r.setdefault("rank_by_e0_delta", -1)
    (OUT / "qualitative").mkdir(exist_ok=True)
    dump("qualitative/candidates_ranked.csv", sorted(cand, key=lambda r: (r["condition"], r["rank_by_e0_delta"])))
    mean = lambda v: float(np.nanmean(v)) if len(v) else float("nan")
    summ = dict(definitions=dict(sky="finite GT disparity == 0", textureless=f"{K}x{K} grey local std < {THR} on the left input of that condition",
                                 frames=len(frames()), total="sum of |d_corr - d_clean| over ALL pixels (RobustSpring averages all pixels); shares over valid-GT pixels are in the CSV",
                                 w="tile-fused fusion weight: confidence-weighted average of the per-tile w_f, same weights as the disparity fusion"),
                selection_rule="per corruption: frames with sky_frac >= 0.05, top-5 by E0 mean |d_corr-d_clean|, choose the median reduction ratio among them",
                selected=sel, regional={}, fusion={}, error_vs_gt={})
    for m in MODELS:
        for c in CONDS[1:]:
            for rn in ("all", "valid", "sky", "nonsky", "textureless", "textured"):
                rs = [r for r in rows if r["model"] == m and r["condition"] == c and r["region"] == rn]
                summ["regional"][f"{m}|{c}|{rn}"] = {k: mean([r[k] for r in rs]) for k in ("pixel_frac", "mean_abs_change", "share_of_total_all_pixels", "mean_clean_disp", "mean_corr_disp")}
    for c in CONDS:
        allw = np.concatenate(wvals[c])
        summ["fusion"][c] = dict(percentiles={str(p): float(np.percentile(allw, p)) for p in (0, 5, 25, 50, 75, 95, 100)}, mean=float(allw.mean()),
                                 frac_lt_0p99=float((allw < .99).mean()), frac_lt_0p95=float((allw < .95).mean()), frac_lt_0p90=float((allw < .90).mean()),
                                 by_region={rn: dict(mean_w=mean([r["mean_w"] for r in wrows if r["condition"] == c and r["region"] == rn]),
                                                     frac_lt_0p99=mean([r["frac_w_lt_0p99"] for r in wrows if r["condition"] == c and r["region"] == rn]),
                                                     mean_abs_st_minus_mono=mean([r["mean_abs_st_minus_mono"] for r in wrows if r["condition"] == c and r["region"] == rn]))
                                            for rn in ("all", "sky", "nonsky", "textureless", "textured")})
    for m in MODELS:
        for c in CONDS:
            for pn in ("fused", "stereo_only", "mono_only"):
                for rn in ("valid", "sky", "nonsky", "textureless", "textured"):
                    rs = [r["abs_err_vs_gt"] for r in erows if r["model"] == m and r["condition"] == c and r["prediction"] == pn and r["region"] == rn]
                    if rs:
                        summ["error_vs_gt"][f"{m}|{c}|{pn}|{rn}"] = mean(rs)
    (OUT / "regional_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(dict(selected=sel, fusion={c: summ["fusion"][c]["percentiles"] for c in CONDS}), indent=1))
    print("STATS DONE")


def cmd_figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summ = json.loads((OUT / "regional_summary.json").read_text())
    sel = dict(summ["selected"])
    if a.override:
        sel.update(dict(kv.split("=") for kv in a.override))
    fr_by_key = {f.key: f for f in frames()}
    (OUT / "qualitative").mkdir(exist_ok=True)
    (OUT / "fusion_maps").mkdir(exist_ok=True)
    rows_fig = []
    for cond, key in sel.items():
        fr = fr_by_key[key]
        item = read_frame(fr)
        Lc, _ = load_pair(fr, cond, item)
        P = {(m, c): load_pred(m, fr, c) for m in MODELS for c in ("clean", cond)}
        vmax = float(np.nanpercentile(np.concatenate([P[(m, "clean")]["pred"].ravel()[::50] for m in MODELS]), 99))
        dE, dF = (np.abs(P[(m, cond)]["pred"] - P[(m, "clean")]["pred"]) for m in ("E0", "Full"))
        dmax = max(1.0, float(np.percentile(dE[::4, ::4], 99)))
        tag = f"{cond}_{fr.scene}_{fr.frame:04d}"
        fig, ax = plt.subplots(2, 4, figsize=(20, 6.2))
        items = [(item["left"], "clean left", None), (Lc, f"{cond} left", None),
                 (P[("E0", "clean")]["pred"], "E0 clean", ("viridis", 0, vmax)), (P[("E0", cond)]["pred"], f"E0 {cond}", ("viridis", 0, vmax)),
                 (dE, f"E0 |Δ| (mean {dE.mean():.2f} px)", ("magma", 0, dmax)), (P[("Full", "clean")]["pred"], "Full clean", ("viridis", 0, vmax)),
                 (P[("Full", cond)]["pred"], f"Full {cond}", ("viridis", 0, vmax)), (dF, f"Full |Δ| (mean {dF.mean():.2f} px)", ("magma", 0, dmax))]
        for axx, (img, title, cm) in zip(ax.ravel(), items):
            h = axx.imshow(img) if cm is None else axx.imshow(img, cmap=cm[0], vmin=cm[1], vmax=cm[2])
            if cm is not None:
                fig.colorbar(h, ax=axx, fraction=0.025, pad=0.01)
            axx.set_title(title, fontsize=11)
            axx.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / "qualitative" / f"panel_{tag}.png", dpi=110)
        plt.close(fig)
        # fusion maps: clean and corrupted input
        fig, ax = plt.subplots(2, 4, figsize=(20, 6.2))
        for r_, c_ in enumerate(("clean", cond)):
            inp = item["left"] if c_ == "clean" else Lc
            w = P[("Full", c_)]["w"]
            dsm = np.abs(P[("Full", c_)]["d_st"] - P[("Full", c_)]["d_mono"])
            ax[r_, 0].imshow(inp); ax[r_, 0].set_title(f"{c_} left", fontsize=11)
            h = ax[r_, 1].imshow(w, cmap="viridis", vmin=min(0.9, float(w.min())), vmax=1.0); fig.colorbar(h, ax=ax[r_, 1], fraction=0.025, pad=0.01)
            ax[r_, 1].set_title(f"w_f (min {w.min():.3f}, mean {w.mean():.4f})", fontsize=11)
            ax[r_, 2].imshow(inp); h = ax[r_, 2].imshow(1 - w, cmap="inferno", alpha=0.6, vmin=0, vmax=max(0.02, float((1 - w).max())))
            fig.colorbar(h, ax=ax[r_, 2], fraction=0.025, pad=0.01); ax[r_, 2].set_title("1 - w_f overlay", fontsize=11)
            h = ax[r_, 3].imshow(dsm, cmap="magma", vmin=0, vmax=float(np.percentile(dsm[::4, ::4], 99))); fig.colorbar(h, ax=ax[r_, 3], fraction=0.025, pad=0.01)
            ax[r_, 3].set_title("|d_st - d_mono| (px)", fontsize=11)
            for x in ax[r_]:
                x.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / "fusion_maps" / f"fusion_{tag}.png", dpi=110)
        plt.close(fig)
        rows_fig.append((cond, Lc, dE, dF, P[("Full", cond)]["w"], dmax))
    # ---- publication candidates
    for name, last in (("figure_qualitative_with_wf", True), ("figure_qualitative", False)):
        ncol = 4 if last else 3
        fig, ax = plt.subplots(len(rows_fig), ncol, figsize=(4.6 * ncol, 2.75 * len(rows_fig)), squeeze=False)
        for i, (cond, Lc, dE, dF, w, dmax) in enumerate(rows_fig):
            ax[i, 0].imshow(Lc); ax[i, 0].set_ylabel(cond)
            h1 = ax[i, 1].imshow(dE, cmap="magma", vmin=0, vmax=dmax)
            h2 = ax[i, 2].imshow(dF, cmap="magma", vmin=0, vmax=dmax)
            fig.colorbar(h2, ax=ax[i, 2], fraction=0.025, pad=0.01)
            ax[i, 1].set_xlabel(f"{dE.mean():.2f} px", fontsize=9); ax[i, 2].set_xlabel(f"{dF.mean():.2f} px", fontsize=9)
            if last:
                h3 = ax[i, 3].imshow(w, cmap="viridis", vmin=min(0.9, float(w.min())), vmax=1.0)
                fig.colorbar(h3, ax=ax[i, 3], fraction=0.025, pad=0.01)
            if i == 0:
                for j, t in enumerate(["Corrupted left", "E0  |Δd|", "Full  |Δd|", "Full  w_f"][:ncol]):
                    ax[i, j].set_title(t, fontsize=11)
            for x in ax[i]:
                x.set_xticks([]); x.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / f"{name}.png", dpi=200)
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)
    print("FIGURES DONE", sel)


def cmd_latency(a):
    model, _ = build_croco_stereo(str(MODELS["Full"]), "cuda")
    model.eval()
    net = model.net
    n_mono = sum(p.numel() for p in net.mono.parameters())
    n_all = sum(p.numel() for p in net.parameters())
    g = torch.Generator(device="cuda").manual_seed(0)
    x1, x2 = (torch.randn(1, 3, 352, 704, device="cuda", generator=g) for _ in range(2))
    L, R = (torch.rand(1, 3, 1080, 1920, device="cuda", generator=g) * 255 for _ in range(2))

    def timeit(fn, warm, n):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        return ts

    def stat(ts):
        t = np.array(ts) * 1000
        return dict(mean_ms=float(t.mean()), median_ms=float(np.median(t)), std_ms=float(t.std(ddof=1)), p95_ms=float(np.percentile(t, 95)), n=len(ts))

    res = dict(device=torch.cuda.get_device_name(0), precision="fp32", batch=1, tile=[352, 704], warmup_tile=20, note="A and B are timed in alternating blocks on the same GPU")
    with torch.no_grad():
        tile = {"stereo_only": [], "full": []}
        frame = {"stereo_only": [], "full": []}
        mem = {}
        for block in range(5):  # interleave A / B so that any background load affects both equally
            with mono_disabled(net):
                torch.cuda.reset_peak_memory_stats()
                tile["stereo_only"] += timeit(lambda: net(x1, x2), 20 if block == 0 else 3, 20)
                mem["stereo_only"] = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
            tile["full"] += timeit(lambda: net(x1, x2), 20 if block == 0 else 3, 20)
            mem["full"] = torch.cuda.max_memory_allocated() / 2**20
        for block in range(4):
            with mono_disabled(net):
                frame["stereo_only"] += timeit(lambda: model.tiled_predict(L, R, overlap=0.7, tile_batch=1), 1, 5)
            frame["full"] += timeit(lambda: model.tiled_predict(L, R, overlap=0.7, tile_batch=1), 1, 5)
    res["per_tile"] = {k: stat(v) for k, v in tile.items()}
    res["full_frame_1080p_tiled"] = {k: stat(v) for k, v in frame.items()}
    res["overhead_percent"] = dict(per_tile_mean=(res["per_tile"]["full"]["mean_ms"] / res["per_tile"]["stereo_only"]["mean_ms"] - 1) * 100,
                                   per_tile_median=(res["per_tile"]["full"]["median_ms"] / res["per_tile"]["stereo_only"]["median_ms"] - 1) * 100,
                                   full_frame_median=(res["full_frame_1080p_tiled"]["full"]["median_ms"] / res["full_frame_1080p_tiled"]["stereo_only"]["median_ms"] - 1) * 100)
    res["peak_cuda_mem_mib_per_tile"] = mem
    res["parameters"] = dict(full=n_all, mono_and_fuser=n_mono, stereo_only=n_all - n_mono, increase_percent=n_mono / (n_all - n_mono) * 100)
    from track2.croco_model import num_tiles
    res["tiles_per_1080p_frame"] = num_tiles(1080, 1920)
    (OUT / "inference_overhead.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["predict", "stats", "figures", "latency"])
    ap.add_argument("--override", nargs="*", default=[], help="figures: cond=frame_key to replace an automatically selected example")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"predict": cmd_predict, "stats": cmd_stats, "figures": cmd_figures, "latency": cmd_latency}[a.cmd](a)

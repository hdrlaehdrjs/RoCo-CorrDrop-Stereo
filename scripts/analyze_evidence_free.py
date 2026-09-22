#!/usr/bin/env python3
"""Where does the residual robustness error live? (analysis only, no training)

For each model and frozen robust20 corruption on NORMAL val scenes: per-pixel |d_corr - d_clean| stratified by
  (a) sky (GT == 0) vs non-sky,
  (b) local texture of the CORRUPTED left image (evidence available or not; 15x15 grey std < thr),
      and "destroyed" = textured in the clean image but textureless in the corrupted one,
  (c) GT disparity range,
plus the mean prediction inside the sky under clean vs corrupted input (hallucination check).
Shares = sum of |delta| inside the stratum / sum over all pixels (the RobustSpring metric averages all pixels).
Outputs: <out>/rows.json, <out>/summary.md, <out>/panels_<frame>.png
"""
from __future__ import annotations

import argparse
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
from track2.croco_model import build_croco_stereo  # noqa: E402
from track2.spring_data import load_split, read_frame  # noqa: E402

R20 = WORK / "outputs/track2/robust20_val"
NORMAL = {"0038", "0039", "0045", "0047"}
DEFAULT_FRAMES = ["0038:0001", "0038:0072", "0039:0026", "0045:0066", "0047:0089", "0047:0222"]
BINS = [(0, 0, "sky(0)"), (0, 2, "(0,2]"), (2, 8, "(2,8]"), (8, 32, "(8,32]"), (32, 1e9, ">32")]


def local_std(img255: np.ndarray, k: int = 15) -> torch.Tensor:
    g = torch.from_numpy(img255).float().cuda().permute(2, 0, 1)[None]
    g = 0.299 * g[:, :1] + 0.587 * g[:, 1:2] + 0.114 * g[:, 2:3]
    pad = k // 2
    m = F.avg_pool2d(F.pad(g, (pad,) * 4, mode="reflect"), k, 1)
    m2 = F.avg_pool2d(F.pad(g * g, (pad,) * 4, mode="reflect"), k, 1)
    return (m2 - m * m).clamp_min(0).sqrt()[0, 0]


def share(diff, mask):
    tot = float(diff.sum())
    return float(diff[mask].sum()) / tot if tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["E0=checkpoints/track2/croco_clean_teacher.pt",
                                                    "Full=outputs/track2/track2_croco_x7_x6b_mono/stageC_seed2026/final.pt"])
    ap.add_argument("--frames", nargs="+", default=DEFAULT_FRAMES)
    ap.add_argument("--corruptions", default="fog,frost,snow,motion_blur,spatter,gaussian_noise")
    ap.add_argument("--thr", type=float, default=2.0, help="grey std below which a 15x15 patch is 'textureless'")
    ap.add_argument("--out", type=Path, default=WORK / "outputs/track2/analysis/evidence_free")
    ap.add_argument("--panel-frame", default="0038:0072")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    conds = a.corruptions.split(",")
    frames = {f.key: f for f in load_split("val")}
    sel = [frames[k] for k in a.frames]
    assert all(f.scene in NORMAL for f in sel)
    rows = []
    panels = {}
    t0 = time.time()
    for spec in a.models:
        name, ck = spec.split("=", 1)
        model, _ = build_croco_stereo(str(ROOT / ck), "cuda")
        model.eval()
        for fr in sel:
            it = read_frame(fr)
            gt = torch.from_numpy(it["disp"]).cuda()
            fin = torch.isfinite(gt)
            sky = fin & (gt == 0)
            tex_clean = local_std(it["left"])
            to = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda()
            with torch.no_grad():
                clean = model.tiled_predict(to(it["left"]), to(it["right"]), overlap=0.7, tile_batch=1)[0, 0]
            for c in conds:
                load = lambda side: np.asarray(Image.open(R20 / c / fr.scene / f"frame_{side}_{fr.frame:04d}.png").convert("RGB")).copy()
                Lc, Rc = load("left"), load("right")
                with torch.no_grad():
                    pred = model.tiled_predict(to(Lc), to(Rc), overlap=0.7, tile_batch=1)[0, 0]
                diff = (pred - clean).abs()
                tex_corr = local_std(Lc)
                free = tex_corr < a.thr
                destroyed = free & (tex_clean >= a.thr)
                r = dict(model=name, frame=fr.key, condition=c, delta=float(diff.mean()), delta_p95=float(diff.flatten()[::13].quantile(.95)),
                         frac_sky=float(sky.float().mean()), frac_free=float(free.float().mean()), frac_destroyed=float(destroyed.float().mean()),
                         share_sky=share(diff, sky), share_nonsky=share(diff, fin & ~sky), share_free=share(diff, free),
                         share_destroyed=share(diff, destroyed), share_textured=share(diff, ~free),
                         delta_sky=float(diff[sky].mean()) if sky.any() else float("nan"), delta_nonsky=float(diff[fin & ~sky].mean()),
                         delta_free=float(diff[free].mean()) if free.any() else float("nan"), delta_textured=float(diff[~free].mean()),
                         pred_clean_sky=float(clean[sky].mean()) if sky.any() else float("nan"),
                         pred_corr_sky=float(pred[sky].mean()) if sky.any() else float("nan"))
                for lo, hi, lab in BINS:
                    m = fin & ((gt == 0) if hi == 0 else ((gt > lo) & (gt <= hi)))
                    r[f"share_bin{lab}"] = share(diff, m)
                    r[f"frac_bin{lab}"] = float(m.float().mean())
                rows.append(r)
                if fr.key == a.panel_frame and c in ("fog", "frost", "snow"):
                    panels[(name, c)] = dict(img=Lc, clean=clean.cpu().numpy(), corr=pred.cpu().numpy(), free=free.cpu().numpy())
            print(f"[{name}] {fr.key} done {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    (a.out / "rows.json").write_text(json.dumps(rows, indent=1))

    # ---- summary table (mean over frames)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["model"], r["condition"])].append(r)
    keys = ["delta", "share_sky", "share_free", "share_destroyed", "frac_sky", "frac_free", "frac_destroyed", "delta_sky", "delta_nonsky",
            "delta_free", "delta_textured", "pred_clean_sky", "pred_corr_sky"] + [f"share_bin{lab}" for _, _, lab in BINS]
    mean = lambda rs, k: float(np.nanmean([x[k] for x in rs]))
    md = [f"# Evidence-free stratification (normal val scenes, {len(sel)} frames, thr={a.thr})", "",
          "Shares = fraction of the summed |d_corr - d_clean| falling in the stratum (metric averages ALL pixels).", "",
          "| model | corruption | dAbs | share sky | share textureless | share destroyed | frac sky | frac textureless | frac destroyed | dAbs sky | dAbs non-sky | dAbs textureless | dAbs textured | pred sky clean | pred sky corr |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m_, c) in sorted(agg):
        rs = agg[(m_, c)]
        md.append(f"| {m_} | {c} | {mean(rs,'delta'):.2f} | {mean(rs,'share_sky'):.2f} | {mean(rs,'share_free'):.2f} | {mean(rs,'share_destroyed'):.2f} | "
                  f"{mean(rs,'frac_sky'):.2f} | {mean(rs,'frac_free'):.2f} | {mean(rs,'frac_destroyed'):.2f} | {mean(rs,'delta_sky'):.2f} | {mean(rs,'delta_nonsky'):.2f} | "
                  f"{mean(rs,'delta_free'):.2f} | {mean(rs,'delta_textured'):.2f} | {mean(rs,'pred_clean_sky'):.2f} | {mean(rs,'pred_corr_sky'):.2f} |")
    md += ["", "## Share of |delta| by GT disparity bin", "", "| model | corruption | " + " | ".join(lab for _, _, lab in BINS) + " |",
           "|---|---|" + "---|" * len(BINS)]
    for (m_, c) in sorted(agg):
        rs = agg[(m_, c)]
        md.append(f"| {m_} | {c} | " + " | ".join(f"{mean(rs, f'share_bin{lab}'):.2f} (px {mean(rs, f'frac_bin{lab}'):.2f})" for _, _, lab in BINS) + " |")
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- panels
    if panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        models = sorted({k[0] for k in panels})
        conds3 = [c for c in ("fog", "frost", "snow") if any(k[1] == c for k in panels)]
        fig, axes = plt.subplots(len(conds3) * len(models), 4, figsize=(20, 3.2 * len(conds3) * len(models)))
        axes = np.atleast_2d(axes)
        i = 0
        for c in conds3:
            for m_ in models:
                p = panels[(m_, c)]
                vmax = np.nanpercentile(p["clean"], 99)
                d = np.abs(p["corr"] - p["clean"])
                axes[i, 0].imshow(p["img"]); axes[i, 0].set_title(f"{c} left ({m_})")
                axes[i, 1].imshow(p["clean"], vmin=0, vmax=vmax); axes[i, 1].set_title("d_clean")
                axes[i, 2].imshow(p["corr"], vmin=0, vmax=vmax); axes[i, 2].set_title("d_corr")
                axes[i, 3].imshow(d, vmin=0, vmax=np.nanpercentile(d, 99)); axes[i, 3].contour(p["free"], levels=[0.5], colors="r", linewidths=0.5)
                axes[i, 3].set_title(f"|d_corr - d_clean| mean {d.mean():.2f} (red: textureless)")
                for ax in axes[i]:
                    ax.axis("off")
                i += 1
        fig.tight_layout()
        fig.savefig(a.out / f"panels_{a.panel_frame.replace(':', '_')}.png", dpi=80)
    print("DONE", a.out)


if __name__ == "__main__":
    main()

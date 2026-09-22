#!/usr/bin/env python3
"""Appendix qualitative figure: clean | corrupted | E0 |dd| | Full |dd| for one fog / frost / snow example.

Uses ONLY existing outputs (no inference, no training):
  predictions : outputs/track2/final_analysis/cache/{E0,Full}_<scene>_<frame>_<cond>.npz   (key 'pred', full-res disparity)
  inputs      : Spring validation left images (clean) and outputs/track2/robust20_val/<cond>/<scene>/frame_left_<frame>.png
  ranking     : outputs/track2/final_analysis/qualitative/candidates_ranked.csv, regional_stats.csv
Drift map = |d_corrupted - d_clean| of the SAME model (not a GT error). Sky = finite GT disparity == 0.

  python3 scripts/figures/qualitative_robustness_appendix.py candidates      # ranked list + contact sheets
  python3 scripts/figures/qualitative_robustness_appendix.py figure [--pick fog=0038:0072 ...] [--vmax 60]
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import final_analysis as FA  # noqa: E402  (reuses frames(), load_pair(), load_pred(), read_frame, R20, OUT)

OUT = FA.OUT / "appendix_qualitative"
CONDS = ("fog", "frost", "snow")
DEFAULT = {"fog": "0038:0072", "frost": "0045:0066", "snow": "0047:0222"}
NOTE = {"fog": "sky / low-texture drift strongly reduced", "frost": "frost-induced drift substantially reduced",
        "snow": "sky and low-evidence drift substantially reduced"}


def table():
    reg = {}
    for r in csv.DictReader(open(FA.OUT / "regional_stats.csv")):
        reg[(r["model"], r["frame"], r["condition"], r["region"])] = float(r["mean_abs_change"])
    rows = list(csv.DictReader(open(FA.OUT / "qualitative/candidates_ranked.csv")))
    for r in rows:
        for m in ("E0", "Full"):
            r[f"sky_{m}"] = reg.get((m, r["frame"], r["condition"], "sky"), float("nan"))
    return rows


def item_of(frame_id):
    fr = next(f for f in FA.frames() if f"{f.scene}:{f.frame:04d}" == frame_id)
    return fr, FA.read_frame(fr)


def drift(name, fr, cond):
    return np.abs(FA.load_pred(name, fr, cond)["pred"] - FA.load_pred(name, fr, "clean")["pred"])


def cmd_candidates(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = table()
    for c in CONDS:
        cs = sorted([r for r in rows if r["condition"] == c and float(r["sky_frac"]) >= 0.05], key=lambda r: -float(r["e0_delta"]))[:6]
        print(f"== {c}: top-6 by E0 mean |dd| among frames with sky >= 5 % (28-frame primary subset)")
        fig, ax = plt.subplots(len(cs), 3, figsize=(12, 2.3 * len(cs)))
        for i, r in enumerate(cs):
            print(f"  {r['frame']}  E0 {float(r['e0_delta']):7.2f} -> Full {float(r['full_delta']):6.2f} px | sky {float(r['sky_frac']):.2f} "
                  f"| sky-region E0 {r['sky_E0']:7.2f} -> Full {r['sky_Full']:6.2f} | auto-selected={r['selected']}")
            fr, item = item_of(r["frame"])
            e0, fu = drift("E0", fr, c), drift("Full", fr, c)
            vm = np.percentile(e0, 99)
            ax[i, 0].imshow(FA.load_pair(fr, c, item)[0][::4, ::4]); ax[i, 0].set_ylabel(r["frame"])
            ax[i, 1].imshow(e0[::4, ::4], cmap="magma", vmin=0, vmax=vm); ax[i, 2].imshow(fu[::4, ::4], cmap="magma", vmin=0, vmax=vm)
            for x in ax[i]:
                x.set_xticks([]); x.set_yticks([])
        fig.tight_layout(); fig.savefig(OUT / f"candidates_{c}.png", dpi=80); plt.close(fig)


def cmd_figure(a):
    OUT.mkdir(parents=True, exist_ok=True)
    pick = dict(DEFAULT)
    pick.update(dict(p.split("=") for p in a.pick))
    rows = {(r["frame"], r["condition"]): r for r in table()}
    data = []
    for c in CONDS:
        fr, item = item_of(pick[c])
        data.append(dict(cond=c, id=pick[c], clean=item["left"], corr=FA.load_pair(fr, c, item)[0],
                         e0=drift("E0", fr, c), full=drift("Full", fr, c), r=rows[(pick[c], c)]))
    vmax = a.vmax
    fig = plt.figure(figsize=(13.2, 5.72))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.035], wspace=0.025, hspace=0.035, left=0.032, right=0.945, top=0.94, bottom=0.03)
    titles = ["Clean", "Corrupted", r"E0 $|\Delta d|$", r"Full $|\Delta d|$"]
    im = None
    for i, d in enumerate(data):
        r = d["r"]
        for j, img in enumerate((d["clean"], d["corr"], d["e0"], d["full"])):
            ax = fig.add_subplot(gs[i, j])
            if j < 2:
                ax.imshow(img[::2, ::2], interpolation="antialiased")
            else:
                im = ax.imshow(img[::2, ::2], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
                m = "E0" if j == 2 else "Full"
                ax.text(0.985, 0.04, f"mean {float(r['e0_delta' if j == 2 else 'full_delta']):.1f} px | sky {r['sky_' + m]:.1f} px",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.5, edgecolor="none"))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=14.5, pad=5)
            if j == 0:
                ax.set_ylabel(d["cond"].capitalize(), fontsize=14)   # scene / frame / severity live in the caption
    cax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.ax.tick_params(labelsize=12)
    cb.set_label(r"$|\Delta d| = |d_{corr} - d_{clean}|$ (px), clipped at %g" % vmax, fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"qualitative_robustness_appendix.{ext}", dpi=300 if ext == "png" else 200)
    plt.close(fig)
    for d in data:
        r = d["r"]
        print(f"{d['cond']:5s} {d['id']}  E0 {float(r['e0_delta']):.2f} -> Full {float(r['full_delta']):.2f} | sky {r['sky_E0']:.2f} -> {r['sky_Full']:.2f} "
              f"| clipped px: E0 {(d['e0'] > vmax).mean() * 100:.1f} %, Full {(d['full'] > vmax).mean() * 100:.1f} %")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("candidates")
    f = s.add_parser("figure"); f.add_argument("--pick", nargs="*", default=[]); f.add_argument("--vmax", type=float, default=50.0)
    a = p.parse_args()
    dict(candidates=cmd_candidates, figure=cmd_figure)[a.cmd](a)

#!/usr/bin/env python3
"""Regional (sky / textured) drift statistics of the clean-only baseline and the full model for every seed.

  CUDA_VISIBLE_DEVICES=<gpu> python3 scripts/three_seed/regional.py predict e0:2026 full:2026 [...]
        inference only (final checkpoints, production tiled inference) on the 28 primary frames x {clean, fog, frost,
        snow}; writes <out>/rows_<variant>_s<seed>.csv with per-frame statistics (no prediction is stored)
  python3 scripts/three_seed/regional.py summarize
        <out>/regional_3seed.{md,csv,json} from the six rows files

<out> defaults to outputs/track2/three_seed/regional ($ROCO_REGIONAL_OUT overrides it). Region definitions, frames and
the frozen robust20 cache are those of scripts/final_analysis.py: sky = finite GT disparity == 0, textureless = 15x15
grey local std < 2 on the left input of that condition. Drift = |d_corr - d_clean| of the same checkpoint."""
import csv
import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

import common as K

sys.path.insert(0, str(K.ROOT / "scripts"))
import final_analysis as FA  # noqa: E402

OUT = Path(os.environ.get("ROCO_REGIONAL_OUT", K.OUT / "regional"))
REGIONS = ("all", "sky", "nonsky", "textureless", "textured")


def predict(jobs):
    import torch
    from track2.croco_model import build_croco_stereo
    from track2.spring_data import read_frame
    OUT.mkdir(parents=True, exist_ok=True)
    to = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().cuda()
    frames = FA.frames()
    assert len(frames) == K.N_PRIMARY
    for job in jobs:
        vk, seed = job.split(":")
        seed = int(seed)
        dst = OUT / f"rows_{vk}_s{seed}.csv"
        if dst.exists():
            print(f"[{job}] exists, skipped", flush=True)
            continue
        ck = K.run(vk, seed)["train"] / "final.pt"
        model, _ = build_croco_stereo(str(ck), "cuda")
        model.eval()
        rows = []
        for i, fr in enumerate(frames):
            item = read_frame(fr)
            pred = {}
            for cond in FA.CONDS:
                L, R = FA.load_pair(fr, cond, item)
                with torch.no_grad():
                    d = model.tiled_predict(to(L), to(R), overlap=0.7, tile_batch=1)[0, 0]
                assert bool(torch.isfinite(d).all()), (job, fr.key, cond)
                pred[cond] = d.float().cpu().numpy()
            for cond in FA.CONDS[1:]:
                L, _ = FA.load_pair(fr, cond, item)
                reg = FA.regions(item["disp"], L)
                diff = np.abs(pred[cond] - pred["clean"])
                tot = float(diff.sum())
                for rn in REGIONS:
                    m = reg[rn]
                    n = int(m.sum())
                    rows.append(dict(variant=vk, seed=seed, frame=fr.key, condition=cond, region=rn, pixel_frac=n / m.size,
                                     mean_abs_change=float(diff[m].mean()) if n else float("nan"),
                                     share_of_total=float(diff[m].sum()) / tot if tot > 0 else float("nan")))
            print(f"[{job}] {i + 1}/{len(frames)} {fr.key}", flush=True)
        with dst.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        (OUT / f"rows_{vk}_s{seed}.sha256").write_text(K.sha256(ck) + "  final.pt\n")
        del model
        torch.cuda.empty_cache()
    print("PREDICT DONE", flush=True)


def load(vk, seed):
    rs = list(csv.DictReader((OUT / f"rows_{vk}_s{seed}.csv").open()))
    assert len(rs) == K.N_PRIMARY * 3 * len(REGIONS), (vk, seed, len(rs))
    return {(r["frame"], r["condition"], r["region"]): {k: float(r[k]) for k in ("pixel_frac", "mean_abs_change", "share_of_total")} for r in rs}


def summarize():
    nm = lambda v: float(np.nanmean(v))
    S = {}
    for seed in K.SEEDS:
        e0, fu = load("e0", seed), load("full", seed)
        frames = sorted({k[0] for k in e0})
        s = dict(sky_pixel_frac=nm([e0[(f, "fog", "sky")]["pixel_frac"] for f in frames]))
        for c in FA.CONDS[1:]:
            for vk, d in (("e0", e0), ("full", fu)):
                for rn in REGIONS:
                    s[f"{vk}|{c}|{rn}"] = nm([d[(f, c, rn)]["mean_abs_change"] for f in frames])
                s[f"{vk}|{c}|sky_share"] = nm([d[(f, c, "sky")]["share_of_total"] for f in frames])
            worse = [f for f in frames if fu[(f, c, "all")]["mean_abs_change"] > e0[(f, c, "all")]["mean_abs_change"]]
            s[f"n_full_worse|{c}"] = len(worse)
            s[f"max_regression|{c}"] = max([fu[(f, c, "all")]["mean_abs_change"] - e0[(f, c, "all")]["mean_abs_change"] for f in worse], default=0.0)
            s[f"max_sky_frac_of_worse|{c}"] = max([e0[(f, c, "sky")]["pixel_frac"] for f in worse], default=0.0)
            w = max(frames, key=lambda f: e0[(f, c, "all")]["mean_abs_change"])
            s[f"worst_e0_frame|{c}"] = w
            s[f"worst_e0|{c}"] = e0[(w, c, "all")]["mean_abs_change"]
            s[f"full_on_worst_e0_frame|{c}"] = fu[(w, c, "all")]["mean_abs_change"]
            w2 = max(frames, key=lambda f: fu[(f, c, "all")]["mean_abs_change"])
            s[f"worst_full_frame|{c}"] = w2
            s[f"worst_full|{c}"] = fu[(w2, c, "all")]["mean_abs_change"]
        S[seed] = s
    (OUT / "regional_3seed.json").write_text(json.dumps({str(k): v for k, v in S.items()}, indent=1) + "\n")

    def row(label, key, fmt="{:.2f}", scale=1.0):
        v = [S[s][key] * scale for s in K.SEEDS]
        return f"| {label} | " + " | ".join(fmt.format(x) for x in v) + f" | {fmt.format(st.fmean(v))} ± {fmt.format(st.stdev(v))} |"

    L = ["# Regional drift statistics for all three seeds (local development results, 28 primary frames)", "",
         "Inference only. For each seed the clean-only baseline and the full model trained with that seed are compared "
         "(descriptive pairing: the two recipes differ in more than one factor). Drift = |d_corr − d_clean| in px, "
         "mean over the 28 frames of the per-frame regional mean; sky = finite ground-truth disparity 0; "
         "textured = 15×15 grey local std ≥ 2 on the corrupted left image.", "",
         f"Sky pixel fraction (ground truth, identical for all seeds): {100 * S[K.SEEDS[0]]['sky_pixel_frac']:.1f} %", "",
         "| quantity | seed 2026 | seed 2027 | seed 2028 | mean ± sd |", "|---|---|---|---|---|"]
    for c in FA.CONDS[1:]:
        L.append(row(f"{c}: share of baseline drift located in sky (%)", f"e0|{c}|sky_share", "{:.1f}", 100))
        L.append(row(f"{c}: sky drift, clean-only baseline", f"e0|{c}|sky"))
        L.append(row(f"{c}: sky drift, Full", f"full|{c}|sky"))
        red = [100 * (1 - S[s][f"full|{c}|sky"] / S[s][f"e0|{c}|sky"]) for s in K.SEEDS]
        L.append(f"| {c}: sky drift reduction (%) | " + " | ".join(f"{x:.0f}" for x in red) + f" | {st.fmean(red):.0f} ± {st.stdev(red):.0f} |")
        L.append(row(f"{c}: textured-region drift, clean-only baseline", f"e0|{c}|textured", "{:.3f}"))
        L.append(row(f"{c}: textured-region drift, Full", f"full|{c}|textured", "{:.3f}"))
        L.append(row(f"{c}: image-wide drift, clean-only baseline", f"e0|{c}|all"))
        L.append(row(f"{c}: image-wide drift, Full", f"full|{c}|all"))
    L += ["", "## Frame-level accounting", "", "| quantity | seed 2026 | seed 2027 | seed 2028 |", "|---|---|---|---|"]
    for c in FA.CONDS[1:]:
        L.append(f"| {c}: frames (of 28) on which Full drifts more than the baseline | " + " | ".join(str(S[s][f"n_full_worse|{c}"]) for s in K.SEEDS) + " |")
        L.append(f"| {c}: largest such regression (px) | " + " | ".join(f"{S[s][f'max_regression|{c}']:.3f}" for s in K.SEEDS) + " |")
        L.append(f"| {c}: largest sky fraction among those frames (%) | " + " | ".join(f"{100 * S[s][f'max_sky_frac_of_worse|{c}']:.1f}" for s in K.SEEDS) + " |")
        L.append(f"| {c}: worst baseline frame → Full on the same frame (px) | " +
                 " | ".join(f"{S[s][f'worst_e0_frame|{c}']}: {S[s][f'worst_e0|{c}']:.1f} → {S[s][f'full_on_worst_e0_frame|{c}']:.1f}" for s in K.SEEDS) + " |")
        L.append(f"| {c}: worst Full frame (px) | " + " | ".join(f"{S[s][f'worst_full_frame|{c}']}: {S[s][f'worst_full|{c}']:.1f}" for s in K.SEEDS) + " |")
    (OUT / "regional_3seed.md").write_text("\n".join(L) + "\n")
    with (OUT / "regional_3seed.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["quantity"] + [f"seed{s}" for s in K.SEEDS])
        for k in S[K.SEEDS[0]]:
            wr.writerow([k] + [S[s][k] for s in K.SEEDS])
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict(sys.argv[2:])
    else:
        summarize()

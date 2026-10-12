#!/usr/bin/env python3
"""Three-seed tables of the local experiments (CPU only, read-only on the runs).

  python3 scripts/three_seed/summarize.py [--out DIR]        (default: outputs/track2/three_seed)

Every run of scripts/three_seed/runs.json is first checked against its own records (training contract, log, evaluation
rows); only verified runs enter a statistic, and a variant with fewer than three verified runs is labelled incomplete.
Writes runs.{md,csv}, ablations.{md,csv}, corrmask.{md,csv,png}, loss_weights.{md,csv} with two plots,
staged_and_other_variants.{md,csv}, summary.{md,csv} and manifest.json.

Local development results (28 primary frames, frozen robust20 cache), not official RobustSpring scores."""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

import common as K
from track2.config import load_config

MAIN = ("clean_abs", "clean_d1", "dabs20", "fog", "frost", "snow", "motion_blur")
NOTE = ("Local development results: 28 primary frames (scenes 0038/0039/0045/0047) of the scene-disjoint 49-frame validation split, frozen robust20 cache, "
        "final checkpoint. ΔAbs20 = unweighted mean over the 20 corruptions of the mean |d_corr − d_clean|. Not official RobustSpring results. "
        "Seeds 2026 / 2027 / 2028; sd = sample standard deviation (ddof = 1) across independently trained runs, not a standard error or a confidence "
        "interval; three runs do not support significance claims.")


def check_run(r: dict, ref_train: Path, ref: dict | None, r20_sha: str | None) -> tuple[list[str], dict, dict | None]:
    """(problems, facts, metrics) of one run from its own records."""
    tr, ev = r["train"], r["eval"]
    if not (tr / "contract.json").exists():
        return ["training contract not found"], {}, None
    c = json.loads((tr / "contract.json").read_text())
    t = K.train_stats(tr)
    steps = c["config"]["steps"]
    probs = []
    rec, cfg = K.flat(c["config"]), K.flat(load_config(r["config"]))
    if {k for k in set(rec) | set(cfg) if rec.get(k) != cfg.get(k)} - {"config_file"}:
        probs.append("recorded configuration differs from the config file")
    if c["seed"] != r["seed"]:
        probs.append(f"seed {c['seed']}")
    if c["split"]["n_train"] != 4054:
        probs.append("not the 4,054-pair training split")
    if ref and (c["split"]["train_sha256"] != ref["split"]["train_sha256"] or c["split"]["val_sha256"] != ref["split"]["val_sha256"]
                or c["init_checkpoint_sha256"] != ref["init_checkpoint_sha256"] or c["teacher_checkpoint_sha256"] != ref["teacher_checkpoint_sha256"]):
        probs.append("split or initial / teacher checkpoint differs from the seed-2026 run")
    if t.get("status") != "complete" or t.get("steps_done") != steps or t.get("steps_logged") != steps:
        probs.append(f"training incomplete ({t.get('steps_logged')}/{steps})")
    bad = K.nonfinite_steps(tr, ref_train) if (ref_train / "train.jsonl").exists() and (tr / "train.jsonl").exists() else None
    if bad:
        probs.append(f"{bad} non-finite steps")
    facts = dict(steps=t.get("steps_done"), train_hours=t.get("wall_hours"), peak_gib=t.get("peak_gib"), gpu=t.get("gpu"),
                 corrmask_event_frac=t.get("corrmask_event_frac"), aug_frac=t.get("aug_frac"), nonfinite_steps=bad)
    m, msg = K.load_eval(ev)
    if m is None:
        return probs + [f"evaluation: {msg}"], facts, None
    ec = json.loads((ev / "contract.json").read_text())
    facts["checkpoint_sha256"] = ec["checkpoint_sha256"]
    if (tr / "final.pt").exists():
        facts["checkpoint"] = "present"
        if K.sha256(tr / "final.pt") != ec["checkpoint_sha256"]:
            probs.append("evaluated checkpoint is not the final checkpoint")
    elif (tr / "WEIGHTS_DELETED.txt").exists() and ec["checkpoint_sha256"] in (tr / "WEIGHTS_DELETED.txt").read_text():
        facts["checkpoint"] = "deleted after evaluation (sha256 recorded)"
    else:
        probs.append("final checkpoint missing")
    if r20_sha and ec["suite"]["manifest_sha256"] != r20_sha:
        probs.append("evaluation not on the frozen robust20 cache")
    facts["robust20_manifest_sha256"] = ec["suite"]["manifest_sha256"]
    return probs, facts, m


def gather():
    r20_sha = K.sha256(K.R20 / "manifest.json") if (K.R20 / "manifest.json").exists() else None
    rows, data = [], {}
    for vk, v in K.VARIANTS.items():
        ref_run = K.run(vk, 2026)
        ref = json.loads((ref_run["train"] / "contract.json").read_text()) if (ref_run["train"] / "contract.json").exists() else None
        data[vk] = {}
        for seed in K.SEEDS:
            r = K.run(vk, seed)
            probs, facts, m = check_run(r, ref_run["train"], ref, r20_sha)
            rows.append(dict(variant=vk, label=v["label"], seed=seed, config=r["config"], status="verified" if not probs else "; ".join(probs),
                             train=K.rel(r["train"]), eval=K.rel(r["eval"]), **facts, **({k: (m[k] if k in m else m["per_corruption"][k]) for k in MAIN} if m else {}),
                             **({f"dabs_{c}": m["per_corruption"][c] for c in K.CORRUPTIONS20} if m else {}),
                             **({"all49_clean_abs": m["all49_clean_abs"], "all49_dabs20": m["all49_dabs20"]} if m else {})))
            if m and not probs:
                pc = m["per_corruption"]
                data[vk][seed] = dict(m, fog=pc["fog"], frost=pc["frost"], snow=pc["snow"], motion_blur=pc["motion_blur"], facts=facts)
    return rows, data


def log_mean(train_dir: Path, keys, last: int = 0) -> dict:
    """Mean of logged per-step quantities of one run (over the steps that log them; `last` > 0: only the last steps)."""
    rows = [json.loads(l) for l in (train_dir / "train.jsonl").read_text().splitlines() if l.strip()][-last:]
    out = {}
    for k in keys:
        v = [x[k] for x in rows if isinstance(x.get(k), (int, float)) and x[k] == x[k]]
        out[k] = (st.fmean(v), len(v)) if v else (float("nan"), 0)
    return out


def ms(d, k):
    xs = [d[s][k] for s in K.SEEDS if s in d]
    return (st.fmean(xs), st.stdev(xs)) if len(xs) == 3 else (st.fmean(xs) if xs else float("nan"), float("nan"))


def cell(d, k, p=3):
    if not d:
        return "–"
    m, s = ms(d, k)
    return f"{m:.{p}f} ± {s:.{p}f}" if len(d) == 3 else f"{m:.{p}f} (incomplete, n={len(d)})"


def indiv(d, k="dabs20", p=3):
    return " / ".join(f"{d[s][k]:.{p}f}" if s in d else "–" for s in K.SEEDS)


def matched(d, full, k="dabs20"):
    diffs = {s: d[s][k] - full[s][k] for s in K.SEEDS if s in d and s in full}
    txt = " / ".join(f"{diffs[s]:+.3f}" if s in diffs else "–" for s in K.SEEDS)
    if len(diffs) == 3:
        xs = list(diffs.values())
        return txt, f"{st.fmean(xs):+.3f} ± {st.stdev(xs):.3f}", sum(x > 0 for x in xs)
    return txt, f"incomplete (n={len(diffs)})", sum(x > 0 for x in diffs.values())


def table(data, keys, title):
    full = data["full"]
    L = [f"## {title}", "", "| variant | ΔAbs20 (2026 / 2027 / 2028) | mean ± sd | matched-seed Δ vs Full (2026 / 2027 / 2028) | mean ± sd of Δ | seeds worse than Full | Clean Abs | Clean D1 | Fog | Frost | Snow | Motion blur |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for vk in keys:
        d, v = data[vk], K.VARIANTS[vk]
        tail = f"{cell(d, 'clean_abs', 4)} | {cell(d, 'clean_d1')} | {cell(d, 'fog', 2)} | {cell(d, 'frost', 2)} | {cell(d, 'snow', 2)} | {cell(d, 'motion_blur', 2)} |"
        if vk == "full":
            L.append(f"| **{v['label']}** | {indiv(d)} | {cell(d, 'dabs20')} | – | – | – | {tail}")
        elif v["one_factor_ablation_of_full"]:
            mt, mm, nw = matched(d, full)
            L.append(f"| {v['label']} | {indiv(d)} | {cell(d, 'dabs20')} | {mt} | {mm} | {nw} of {len(d)} | {tail}")
        else:
            L.append(f"| {v['label']} | {indiv(d)} | {cell(d, 'dabs20')} | (different recipe) | – | – | {tail}")
    return L


def percorr(data, keys):
    L = ["", "### Per-corruption ΔAbs (mean over the verified seeds; n in the header)", "", "| corruption | " + " | ".join(f"{K.VARIANTS[k]['label']} (n={len(data[k])})" for k in keys) + " |", "|---|" + "---|" * len(keys)]
    for c in K.CORRUPTIONS20:
        L.append(f"| {c} | " + " | ".join((f"{st.fmean(data[k][s]['per_corruption'][c] for s in data[k]):.3f}" if data[k] else "–") for k in keys) + " |")
    return L


def clean_matched(data, keys):
    full = data["full"]
    L = ["", "### Clean accuracy, matched seeds (variant − Full)", "", "| variant | Clean Abs Δ in px (2026 / 2027 / 2028) | mean ± sd | Clean D1 Δ in points (2026 / 2027 / 2028) |", "|---|---|---|---|"]
    for vk in keys:
        if vk == "full" or not K.VARIANTS[vk]["one_factor_ablation_of_full"]:
            continue
        d = data[vk]
        da = {s: d[s]["clean_abs"] - full[s]["clean_abs"] for s in K.SEEDS if s in d and s in full}
        txt = " / ".join((f"{da[s]:+.4f}" if s in da else "–") for s in K.SEEDS)
        mm = f"{st.fmean(da.values()):+.4f} ± {st.stdev(da.values()):.4f}" if len(da) == 3 else f"incomplete (n={len(da)})"
        L.append(f"| {K.VARIANTS[vk]['label']} | {txt} | {mm} | {matched(d, full, 'clean_d1')[0]} |")
    return L


def write_csv(path, data, keys):
    full = data["full"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "label", "seed"] + list(MAIN) + ["dabs20_minus_full_same_seed", "clean_abs_minus_full_same_seed"] + [f"dabs_{c}" for c in K.CORRUPTIONS20])
        for vk in keys:
            d, v = data[vk], K.VARIANTS[vk]
            ctrl = v["one_factor_ablation_of_full"]
            for s in K.SEEDS:
                if s not in d:
                    w.writerow([vk, v["label"], s, "MISSING"])
                    continue
                r = d[s]
                w.writerow([vk, v["label"], s] + [f"{r[k]:.5f}" for k in MAIN]
                           + [f"{r['dabs20'] - full[s]['dabs20']:+.5f}" if ctrl and s in full else "", f"{r['clean_abs'] - full[s]['clean_abs']:+.5f}" if ctrl and s in full else ""]
                           + [f"{r['per_corruption'][c]:.5f}" for c in K.CORRUPTIONS20])
            for nm, i in (("mean", 0), ("sd", 1)):
                w.writerow([vk, v["label"], nm if len(d) == 3 else f"{nm}_incomplete_n{len(d)}"] + [(f"{ms(d, k)[i]:.5f}" if d else "") for k in MAIN])


def plot(path, data, points, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for a_, k, yl in ((ax[0], "dabs20", "ΔAbs20 (px, lower is better)"), (ax[1], "clean_abs", "Clean Abs (px)")):
        xs, mu = [], []
        for x, vk in points:
            ys = [data[vk][s][k] for s in K.SEEDS if s in data[vk]]
            a_.scatter([x] * len(ys), ys, color="tab:red" if vk == "full" else "tab:blue", s=22, zorder=3)
            if ys:
                xs.append(x)
                mu.append(st.fmean(ys))
        a_.plot(xs, mu, color="gray", lw=1, zorder=2)
        a_.set_xlabel(xlabel)
        a_.set_ylabel(yl)
        a_.set_xticks([x for x, _ in points])
        a_.grid(alpha=.3)
    ax[0].set_title("points: runs, line: mean, red: default setting", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=K.OUT)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    rows, data = gather()
    # ---- run inventory (also the per-run cost)
    L = ["# Runs behind the three-seed tables", "", NOTE, "",
         "`verified`: training finished all steps with the registered seed on the 4,054-pair split, from the same initial / teacher checkpoints as the seed-2026 run "
         "of the variant; the recorded configuration equals the config file; no non-finite step; the evaluation has 1029 unique rows (588 on the primary frames) on the "
         "frozen robust20 cache and was run on the final checkpoint. Wall-clock hours are from a shared server (up to four concurrent runs) and are upper bounds.", "",
         "| variant | seed | config | ΔAbs20 | Clean Abs | steps | train h | peak GiB | checkpoint sha256 | status |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda k, p: f"{r[k]:.{p}f}" if isinstance(r.get(k), float) else "–"
        L.append(f"| {r['label']} | {r['seed']} | `{r['config']}` | {f('dabs20', 4)} | {f('clean_abs', 4)} | {r.get('steps', '–')} | {f('train_hours', 1)} | {f('peak_gib', 1)} | {str(r.get('checkpoint_sha256', '–'))[:12]} | {r['status']} |")
    hrs = [r["train_hours"] for r in rows if isinstance(r.get("train_hours"), float)]
    L += ["", f"{sum(r['status'] == 'verified' for r in rows)} of {len(rows)} runs verified. Total training time of the listed runs: {sum(hrs):.0f} GPU-hours "
          f"(median {st.median(hrs):.1f} h per run); evaluation of one checkpoint on the robust20 suite takes about one further GPU-hour." if hrs else ""]
    (out / "runs.md").write_text("\n".join(L) + "\n")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # ---- core ablations
    abl = ["full", "no_dropout", "randkey", "no_delta", "image_plane", "no_teacher", "no_mono", "no_overlay"]
    L = ["# Core ablations over three seeds", "", NOTE, ""] + table(data, abl, "Three-seed summary") + clean_matched(data, abl)
    L += ["", "### Random-key control: masking budget (from each run's training log)", "",
          "Per selected query, the control masks as many right-view keys as the structured epipolar mask would, drawn uniformly without replacement.", "",
          "| seed | steps with dropout | keys of the structured mask | keys masked by the control | of them inside the epipolar band | share of the 132-key band budget |", "|---|---|---|---|---|---|"]
    for s in K.SEEDS:
        if s in data["randkey"]:
            g = log_mean(K.run("randkey", s)["train"], ("drop_keys_struct", "drop_keys_actual", "drop_keys_in_band"))
            L.append(f"| {s} | {g['drop_keys_actual'][1]} | {g['drop_keys_struct'][0]:.1f} | {g['drop_keys_actual'][0]:.1f} | {g['drop_keys_in_band'][0]:.2f} | {100 * g['drop_keys_in_band'][0] / 132:.1f} % |")
    L += ["", "Expected under uniform sampling: 132 / 968 = 13.6 %."] + percorr(data, abl)
    (out / "ablations.md").write_text("\n".join(L) + "\n")
    write_csv(out / "ablations.csv", data, abl)
    # ---- CorrMask probability
    cm = ["cm_p000", "cm_p010", "full", "cm_p050"]
    L = ["# CorrMask sampling probability over three seeds", "", NOTE, "",
         "`s2aug.p_mask` is the CorrMask probability inside the S2Aug gate (p_aug = 0.7 after the 10 % curriculum): nominal unconditional frequency 0.7·p_mask. "
         "The full model is the p_mask = 0.25 row. Structured correspondence dropout (decoder cross-attention, p = 0.5) is unchanged in every row.", ""]
    L += table(data, cm, "Three-seed summary")
    L += ["", "### Observed CorrMask event frequency (from each run's training log)", "", "| setting | seed | events / all training samples | events / augmented samples | S2Aug fraction |", "|---|---|---|---|---|"]
    for vk in cm:
        for s in K.SEEDS:
            if s in data[vk]:
                t = data[vk][s]["facts"]
                L.append(f"| {K.VARIANTS[vk]['label'] if vk != 'full' else 'Full (p_mask = 0.25)'} | {s} | {t['corrmask_event_frac']:.4f} | {t['corrmask_event_frac'] / t['aug_frac']:.4f} | {t['aug_frac']:.4f} |")
    L += clean_matched(data, cm) + percorr(data, cm)
    plot(out / "corrmask.png", data, [(0.0, "cm_p000"), (0.10, "cm_p010"), (0.25, "full"), (0.50, "cm_p050")], "CorrMask probability inside the S2Aug gate (p_mask)")
    (out / "corrmask.md").write_text("\n".join(L + ["", "Plot: `corrmask.png`."]) + "\n")
    write_csv(out / "corrmask.csv", data, cm)
    # ---- loss weights
    ld, lc = ["no_delta", "ld050", "full", "ld200"], ["lc0125", "full", "lc050"]
    L = ["# Loss-weight sensitivity over three seeds", "", NOTE, "",
         "Objective (`scripts/train_track2.py`): `L = λ_clean·L_clean + λ_anchor·L_anchor + λ_mono·L_mono(clean) + η(t)·[λ_corr·L_corr + λ_Δ·L_Δ] + λ_mono·L_mono(corrupted)`, "
         "η(t) = min(1, t/(0.1·T)). Default (λ_clean, λ_anchor, λ_corr, λ_Δ, λ_mono) = (1, 0.1, 0.25, 1, 0.5). One coefficient differs per row; `- L_delta` is the "
         "λ_Δ = 0 ablation and Full is the λ_Δ = 1.0 / λ_corr = 0.25 row.", ""]
    L += table(data, ld, "Consistency weight λ_Δ (0 / 0.5 / 1.0 / 2.0)") + [""] + table(data, lc, "Corrupted-supervision weight λ_corr (0.125 / 0.25 / 0.5)")
    L += clean_matched(data, ["full", "no_delta", "ld050", "ld200", "lc0125", "lc050"]) + percorr(data, ["no_delta", "ld050", "full", "ld200", "lc0125", "lc050"])
    plot(out / "loss_lambda_delta.png", data, [(0.0, "no_delta"), (0.5, "ld050"), (1.0, "full"), (2.0, "ld200")], "λ_Δ")
    plot(out / "loss_lambda_corr.png", data, [(0.125, "lc0125"), (0.25, "full"), (0.5, "lc050")], "λ_corr")
    (out / "loss_weights.md").write_text("\n".join(L + ["", "Plots: `loss_lambda_delta.png`, `loss_lambda_corr.png`."]) + "\n")
    write_csv(out / "loss_weights.csv", data, ["no_delta", "ld050", "full", "ld200", "lc0125", "lc050"])
    # ---- staged variants and exploratory variants
    sv = ["e0", "x4", "no_mono", "full", "x2", "x6a", "x5b", "x9"]
    L = ["# Staged variants and additional variants over three seeds", "", NOTE, "",
         "These rows are separate recipes, not one-factor ablations of the full model: the clean-only baseline uses no teacher, the short fine-tuning starts from the "
         "seed-2026 clean-only checkpoint and trains decoder + head for 1,351 steps, the consistency stage, reliability gating and the attention-supervision variants have "
         "no monocular branch. Every robust variant uses the seed-2026 clean-only checkpoint as its frozen teacher for every seed. "
         "`- monocular branch` is the staged row `+ corr. dropout + overlays`.", ""]
    L += table(data, sv, "Three-seed summary")
    L += ["", "### All 49 validation frames (secondary)", "", "| variant | Clean Abs (mean) | ΔAbs20 (2026 / 2027 / 2028) | mean |", "|---|---|---|---|"]
    for vk in ("e0", "x4", "no_mono", "full"):
        d = data[vk]
        L.append(f"| {K.VARIANTS[vk]['label']} | {cell(d, 'all49_clean_abs').split(' ±')[0]} | {indiv(d, 'all49_dabs20')} | {cell(d, 'all49_dabs20').split(' ±')[0]} |")
    L += ["", "### Reliability gating: logged gate value (mean over the last quarter of training)", "", "| seed | clean branch | corrupted branch |", "|---|---|---|"]
    for s in K.SEEDS:
        if s in data["x6a"]:
            tr = K.run("x6a", s)["train"]
            g = log_mean(tr, ("g_clean", "g_corr"), last=data["x6a"][s]["facts"]["steps"] // 4)
            L.append(f"| {s} | {g['g_clean'][0]:.4f} | {g['g_corr'][0]:.4f} |")
    L += percorr(data, sv)
    (out / "staged_and_other_variants.md").write_text("\n".join(L) + "\n")
    write_csv(out / "staged_and_other_variants.csv", data, sv)
    # ---- all variants
    allk = list(K.VARIANTS)
    L = ["# All local variants — three-seed summary", "", NOTE, "",
         "| group | variant | verified seeds | ΔAbs20 (2026 / 2027 / 2028) | mean ± sd | Clean Abs mean ± sd |", "|---|---|---|---|---|---|"]
    for vk in allk:
        d = data[vk]
        L.append(f"| {K.VARIANTS[vk]['group']} | {K.VARIANTS[vk]['label']} | {len(d)} / 3 | {indiv(d)} | {cell(d, 'dabs20')} | {cell(d, 'clean_abs', 4)} |")
    L += ["", "## Runs that are not part of the statistics", ""]
    reps = []
    for x in K.REPETITIONS:
        m, _ = K.load_eval(K.OUTPUTS / x["eval"])
        orig = data[x["variant"]].get(x["seed"])
        if m:
            reps.append(dict(x, dabs20=m["dabs20"], clean_abs=m["clean_abs"], clean_d1=m["clean_d1"]))
            L.append(f"- Repetition of `{x['config']}` (seed {x['seed']}; {x['note']}): ΔAbs20 {m['dabs20']:.3f}, Clean Abs {m['clean_abs']:.4f}, Clean D1 {m['clean_d1']:.3f}"
                     + (f". Counted run: ΔAbs20 {orig['dabs20']:.3f}, Clean Abs {orig['clean_abs']:.4f}, Clean D1 {orig['clean_d1']:.3f}." if orig else "."))
    for x in K.ABORTED:
        L.append(f"- `{x['run']}`: {x['note']} (`{x['repeated_as']}`); only the repetition is counted.")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    write_csv(out / "summary.csv", data, allk)
    man = dict(protocol=NOTE, seeds=list(K.SEEDS), variants={}, repetitions=reps, aborted=K.ABORTED)
    for vk in allk:
        v = K.VARIANTS[vk]
        man["variants"][vk] = dict(label=v["label"], group=v["group"], one_factor_ablation_of_full=v["one_factor_ablation_of_full"], complete=len(data[vk]) == 3,
                                   runs={str(r["seed"]): {k: r.get(k) for k in ("config", "status", "train", "eval", "checkpoint_sha256", "checkpoint", "steps", "train_hours",
                                                                                 "peak_gib", "gpu", "robust20_manifest_sha256") + MAIN}
                                         for r in rows if r["variant"] == vk})
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(f"{sum(r['status'] == 'verified' for r in rows)} of {len(rows)} runs verified; variants with three verified seeds: {sum(len(data[k]) == 3 for k in allk)} / {len(allk)}; tables in {out}")
    for r in rows:
        if r["status"] != "verified":
            print("  NOT VERIFIED:", r["variant"], r["seed"], r["status"])


if __name__ == "__main__":
    main()

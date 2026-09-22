#!/usr/bin/env python3
"""Targeted ablation summaries (random-key dropout control, image-plane transport repeat, no-teacher repeat).
Reuses the loader / seed check of final_seed_summary.py (28 primary frames, frozen robust20 cache, unchanged metrics).
Writes random_key_dropout_summary.{csv,md}, structural_repeat_summary.{csv,md}, final_targeted_ablation_summary.md and
per_corruption_targeted.csv into outputs/track2/final_analysis/. Missing runs are listed, never imputed; no
significance test (n = 2-3 seeds): mean, sample sd and matched-seed differences only."""
import csv, json, statistics as st
import final_seed_summary as F

EV, FR, OUT, ROOT = F.EV, F.FR, F.OUT, F.ROOT
RUNS = dict(F.RUNS)
RUNS["random-key dropout (control)"] = [(2026, FR / "randkey_seed2026/eval/robust20", None),
                                        (2027, FR / "randkey_seed2027/eval/robust20", None)]
RUNS["image-plane transport"] = [(2026, EV / "r20_croco_ABLimageplane", "track2_croco_abl_image_plane"),
                                 (2027, FR / "image_plane_seed2027/eval/robust20", None)]
RUNS["no teacher / anchor"] = [(2026, EV / "r20_croco_ABLnoteacher", "track2_croco_abl_no_teacher"),
                               (2027, FR / "no_teacher_seed2027/eval/robust20", None)]
ORDER = ["Full", "no correspondence dropout", "random-key dropout (control)", "no L_delta", "image-plane transport",
         "no teacher / anchor", "no mono head", "no sensor overlay"]
MAIN = ("clean_abs", "clean_d1", "dabs20", "fog", "frost", "snow", "motion_blur")


def load_full(d):
    """Like F.load but with clean D1 and all 20 per-corruption DeltaAbs values."""
    res, msg = F.load(d)
    if res is None:
        return None, msg
    rows = [json.loads(l) for l in (d / "rows.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["frame"].split(":")[0] in F.NORMAL]
    per = {}
    for r in rows:
        per.setdefault(r["condition"], []).append(r)
    res["clean_d1"] = st.fmean(r["d1"] for r in per["clean"])
    res["per_corruption"] = {c: st.fmean(r["delta"] for r in v) for c, v in per.items() if c != "clean"}
    mb = [c for c in res["per_corruption"] if "motion" in c]
    res["motion_blur"] = res["per_corruption"][mb[0]] if mb else float("nan")
    return res, "ok"


def collect():
    table, notes = {}, []
    for var in ORDER:
        table[var] = []
        for seed, d, run in RUNS[var]:
            res, msg = load_full(d)
            cs = F.contract_seed(d, run)
            if cs is not None and cs != seed:
                notes.append(f"SEED MISMATCH {var}: expected {seed}, contract says {cs} -> excluded")
            elif res is None:
                notes.append(f"{var} seed {seed}: {msg} ({d.relative_to(WORK)})")
            else:
                table[var].append(dict(seed=seed, path=str(d.relative_to(WORK)), **res))
    return table, notes


def ms(rs, k):
    v = [r[k] for r in rs]
    return st.fmean(v), (st.stdev(v) if len(v) > 1 else float("nan"))


def cell(rs, k, p=3):
    if not rs:
        return "missing"
    m, s = ms(rs, k)
    return f"{m:.{p}f} +- {s:.{p}f}" if len(rs) > 1 else f"{m:.{p}f} (1 run)"


def table_md(table, variants):
    md = ["| Variant | Seeds | individual dAbs20 | mean +- sd | Clean Abs | Clean D1 | Fog | Frost | Snow | Motion blur |",
          "|---|---:|---|---|---|---|---|---|---|---|"]
    for v in variants:
        rs = table[v]
        ind = " / ".join(f"{r['dabs20']:.3f}" for r in rs) or "missing"
        md.append(f"| {v} | {len(rs)} | {ind} | {cell(rs, 'dabs20')} | {cell(rs, 'clean_abs')} | {cell(rs, 'clean_d1')} | "
                  f"{cell(rs, 'fog', 2)} | {cell(rs, 'frost', 2)} | {cell(rs, 'snow', 2)} | {cell(rs, 'motion_blur', 2)} |")
    return md


def matched(table, var, keys=("dabs20", "clean_abs", "fog", "frost", "snow")):
    full = {r["seed"]: r for r in table["Full"]}
    md = [f"| seed | metric | {var} | Full (same seed) | difference |", "|---|---|---|---|---|"]
    for r in table[var]:
        if r["seed"] in full:
            for k in keys:
                md.append(f"| {r['seed']} | {k} | {r[k]:.3f} | {full[r['seed']][k]:.3f} | {r[k] - full[r['seed']][k]:+.3f} |")
    return md


def write_csv(path, table, variants):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed"] + list(MAIN) + ["source"])
        for v in variants:
            for r in table[v]:
                w.writerow([v, r["seed"]] + [f"{r[k]:.4f}" for k in MAIN] + [r["path"]])
            if table[v]:
                w.writerow([v, "mean"] + [f"{ms(table[v], k)[0]:.4f}" for k in MAIN] + [""])
                w.writerow([v, "sd"] + [f"{ms(table[v], k)[1]:.4f}" for k in MAIN] + [""])


def main():
    table, notes = collect()
    head = ["28 primary validation frames (scenes 0038/0039/0045/0047), frozen robust20 severity-3 cache, final checkpoint of",
            "each run, unchanged metric code. Sample sd (n-1). n = 2-3 seeds: descriptive only, **no significance claim**.", ""]
    miss = ["", "## Missing / excluded runs"] + ([f"- {n}" for n in notes] or ["- none"])

    # --- random-key control
    rk = ["Full", "no correspondence dropout", "random-key dropout (control)"]
    md = ["# Random-key dropout control", ""] + head + table_md(table, rk)
    R = table["random-key dropout (control)"]
    if R:
        m = ms(R, "dabs20")[0]
        fm, nm = ms(table["Full"], "dabs20")[0], ms(table["no correspondence dropout"], "dabs20")[0]
        f2 = [r for r in table["Full"] if r["seed"] in {x["seed"] for x in R}]
        md += ["", "## Differences of means (dAbs20)",
               f"- random-key mean - Full mean (3 seeds) = structured epipolar dropout: {m - fm:+.3f}",
               f"- random-key mean - Full mean over the SAME seeds ({'/'.join(str(r['seed']) for r in f2)}): {m - ms(f2, 'dabs20')[0]:+.3f}" if f2 else "",
               f"- random-key mean - no-dropout mean: {m - nm:+.3f}",
               f"- for scale: Full seed sd = {ms(table['Full'], 'dabs20')[1]:.3f}",
               "", "## Matched-seed comparison vs Full (structured dropout)"] + matched(table, "random-key dropout (control)")
        nd = {r["seed"]: r for r in table["no correspondence dropout"]}
        md += ["", "## Matched-seed comparison vs no dropout", "| seed | random-key | no dropout | difference |", "|---|---|---|---|"]
        md += [f"| {r['seed']} | {r['dabs20']:.3f} | {nd[r['seed']]['dabs20']:.3f} | {r['dabs20'] - nd[r['seed']]['dabs20']:+.3f} |"
               for r in R if r["seed"] in nd]
    (OUT / "random_key_dropout_summary.md").write_text("\n".join(md + miss) + "\n")
    write_csv(OUT / "random_key_dropout_summary.csv", table, rk)

    # --- structural repeats
    sr = ["Full", "image-plane transport", "no teacher / anchor"]
    md2 = ["# Structural repeats: image-plane transport and no teacher / anchor", ""] + head + table_md(table, sr)
    for v in sr[1:]:
        md2 += ["", f"## Matched-seed comparison: {v} vs Full"] + matched(table, v)
    (OUT / "structural_repeat_summary.md").write_text("\n".join(md2 + miss) + "\n")
    write_csv(OUT / "structural_repeat_summary.csv", table, sr)

    # --- combined + per corruption
    md3 = ["# Final targeted ablation summary", ""] + head + table_md(table, ORDER)
    conds = sorted(table["Full"][0]["per_corruption"])
    md3 += ["", "## Per-corruption dAbs (mean over available seeds)", "| corruption | " + " | ".join(ORDER) + " |",
            "|---|" + "---|" * len(ORDER)]
    for c in conds:
        md3.append(f"| {c} | " + " | ".join(
            (f"{st.fmean(r['per_corruption'][c] for r in table[v]):.3f}" if table[v] else "-") for v in ORDER) + " |")
    (OUT / "final_targeted_ablation_summary.md").write_text("\n".join(md3 + miss) + "\n")
    with open(OUT / "per_corruption_targeted.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed"] + conds)
        for v in ORDER:
            for r in table[v]:
                w.writerow([v, r["seed"]] + [f"{r['per_corruption'][c]:.4f}" for c in conds])
    print("\n".join(md3 + miss))


if __name__ == "__main__":
    main()

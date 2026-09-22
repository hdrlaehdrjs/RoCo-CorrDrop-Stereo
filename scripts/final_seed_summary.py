#!/usr/bin/env python3
"""Seed aggregation for the paper tables. Reads robust20 rows.jsonl of every run, keeps the 28
primary (normal-scene) validation frames, and writes outputs/track2/final_analysis/seed_ablation_summary.{csv,md}.
Runs whose evaluation is missing or incomplete are listed as such - never skipped silently, never imputed.
No significance test is made: with n = 2-3 seeds only mean, sample sd and effect sizes are reported."""
import csv, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__import__("os").environ.get("ROCO_TRACK2_WORK", ROOT))   # checkpoints/, outputs/, third_party/croco live here
EV = WORK / "outputs/track2/eval"
FR = WORK / "outputs/track2/final_runs"
OUT = WORK / "outputs/track2/final_analysis"
NORMAL = ("0038", "0039", "0045", "0047")
N_FRAMES, N_COND = 28, 20
# variant -> [(seed, eval dir, training run dir holding contract.json)]
RUNS = {
    "Full": [(2026, EV / "r20_croco_X7", "track2_croco_x7_x6b_mono"),
                  (2027, EV / "r20_croco_ABLseed2", "track2_croco_abl_seed2"),
                  (2028, EV / "r20_croco_ABLseed3", "track2_croco_abl_seed3")],
    "no correspondence dropout": [(2026, EV / "r20_croco_ABLnodropout", "track2_croco_abl_no_dropout"),
                                  (2027, FR / "no_dropout_seed2/eval/robust20", None)],
    "no L_delta": [(2026, EV / "r20_croco_ABLnodelta", "track2_croco_abl_no_delta"),
                   (2027, FR / "no_ldelta_seed2/eval/robust20", None)],
    "no mono head": [(2026, EV / "r20_croco_X6b", "track2_croco_x6b_x4_corrdrop"),
                     (2027, EV / "r20_croco_ABLnomonoS2", "track2_croco_abl_no_mono_seed2")],
    "no sensor overlay": [(2026, EV / "r20_croco_ABLnooverlay", "track2_croco_abl_no_overlay"),
                          (2027, EV / "r20_croco_ABLnooverlayS2", "track2_croco_abl_no_overlay_seed2")],
}
KEYS = ("clean_abs", "dabs20", "fog", "frost", "snow")


def contract_seed(d, run):
    cands = [d.parents[1] / "train/contract.json"] if run is None else \
        list((WORK / "outputs/track2" / run).glob("*/contract.json"))
    for c in cands:
        if c.exists():
            j = json.loads(c.read_text())
            return j.get("seed", j.get("config", {}).get("seed"))
    return None


def load(d):
    p = d / "rows.jsonl"
    if not p.exists():
        return None, "evaluation not available yet"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["frame"].split(":")[0] in NORMAL]
    per = {}
    for r in rows:
        per.setdefault(r["condition"], {})[r["frame"]] = r
    conds = [c for c in per if c != "clean"]
    if len(conds) != N_COND or any(len(per[c]) != N_FRAMES for c in per):
        return None, f"incomplete evaluation ({len(conds)} conditions, frames {sorted({len(v) for v in per.values()})})"
    m = lambda c, k: st.fmean(x[k] for x in per[c].values())
    res = dict(clean_abs=m("clean", "abs"), dabs20=st.fmean(m(c, "delta") for c in conds))
    res.update({c: m(c, "delta") for c in ("fog", "frost", "snow")})
    return res, "ok"


def main():
    table, notes = {}, []
    for var, runs in RUNS.items():
        table[var] = []
        for seed, d, run in runs:
            res, msg = load(d)
            cs = contract_seed(d, run)
            if cs is not None and cs != seed:
                notes.append(f"SEED MISMATCH {var}: expected {seed}, contract.json says {cs} -> run excluded")
                continue
            if res is None:
                notes.append(f"{var} seed {seed}: {msg} ({d.relative_to(WORK)})")
                continue
            table[var].append(dict(seed=seed, seed_verified=cs is not None, path=str(d.relative_to(WORK)), **res))
    full = table["Full"]
    fm = {k: st.fmean(r[k] for r in full) for k in KEYS}
    fs = {k: st.stdev(r[k] for r in full) for k in KEYS}
    lines_csv = [["variant", "seed", "n_seeds"] + list(KEYS) + ["dabs20_minus_full_mean", "dabs20_diff_in_full_sd", "source"]]
    md = ["# Seed ablation summary (28 normal-scene validation frames, robust20 severity-3 cache)", "",
          "Mean +- sample standard deviation (n-1). n = 2-3 seeds: **no statistical significance is claimed**; the last",
          "column expresses the difference to the Full mean in units of the Full seed sd (descriptive effect size only).", "",
          "| variant | seeds | clean Abs | dAbs20 | fog | frost | snow | dAbs20 - Full | in Full sd |", "|---|---|---|---|---|---|---|---|---|"]
    for var, rs in table.items():
        for r in rs:
            lines_csv.append([var, r["seed"], 1] + [f"{r[k]:.4f}" for k in KEYS] +
                             [f"{r['dabs20'] - fm['dabs20']:+.4f}", f"{(r['dabs20'] - fm['dabs20']) / fs['dabs20']:+.2f}", r["path"]])
        if not rs:
            continue
        n = len(rs)
        mean = {k: st.fmean(r[k] for r in rs) for k in KEYS}
        sd = {k: (st.stdev(r[k] for r in rs) if n > 1 else float("nan")) for k in KEYS}
        diff = mean["dabs20"] - fm["dabs20"]
        lines_csv.append([var, "mean", n] + [f"{mean[k]:.4f}" for k in KEYS] + [f"{diff:+.4f}", f"{diff / fs['dabs20']:+.2f}", ""])
        lines_csv.append([var, "sd", n] + [f"{sd[k]:.4f}" for k in KEYS] + ["", "", ""])
        cell = lambda k, p=3: f"{mean[k]:.{p}f} +- {sd[k]:.{p}f}" if n > 1 else f"{mean[k]:.{p}f} (1 seed)"
        seeds = "/".join(str(r["seed"]) for r in rs)
        is_full = var.startswith("Full")
        c_diff = "-" if is_full else "%+.3f" % diff
        c_sd = "-" if is_full else "%+.2f" % (diff / fs["dabs20"])
        md.append(" | ".join(["| " + var, seeds, cell("clean_abs"), cell("dabs20"), cell("fog", 2), cell("frost", 2),
                              cell("snow", 2), c_diff, c_sd + " |"]))
    md += ["", "Per-seed dAbs20: " + "; ".join(f"{v}: " + ", ".join(f"{r['seed']}={r['dabs20']:.3f}" for r in rs) for v, rs in table.items() if rs)]
    md += ["", "## Missing / excluded runs"] + ([f"- {n}" for n in notes] or ["- none"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "seed_ablation_summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(lines_csv)
    (OUT / "seed_ablation_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

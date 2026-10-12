#!/usr/bin/env python3
"""LaTeX bodies of the local-result tables of the paper (Tables 1, 2, 5-8, 10-13, 15), generated from the verified
evaluation rows and from the regional statistics.

  python3 scripts/three_seed/latex_tables.py [--out FILE] [--regional FILE]

Defaults: outputs/track2/three_seed/latex_tables.tex and outputs/track2/three_seed/regional/regional_3seed.json
(scripts/three_seed/regional.py). Table 9 (same-checkpoint monocular ablation) is written by
scripts/three_seed/mono_same_checkpoint.py."""
import argparse
import json
import statistics as st
from pathlib import Path

import common as K

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=K.OUT / "latex_tables.tex")
ap.add_argument("--regional", type=Path, default=K.OUT / "regional/regional_3seed.json")
args = ap.parse_args()

D = {}
for vk in K.VARIANTS:
    D[vk] = {}
    for s in K.SEEDS:
        m, msg = K.load_eval(K.run(vk, s)["eval"])
        assert m, (vk, s, msg)
        D[vk][s] = m
R = json.loads(args.regional.read_text())

val = lambda vk, k: [D[vk][s][k] if k in D[vk][s] else D[vk][s]["per_corruption"][k] for s in K.SEEDS]
mean = lambda vk, k: st.fmean(val(vk, k))
sd = lambda vk, k: st.stdev(val(vk, k))
pm = lambda vk, k="dabs20": f"${mean(vk, k):.3f}\\pm{sd(vk, k):.3f}$"
runs = lambda vk: " / ".join(f"{x:.3f}" for x in val(vk, "dabs20"))
dmean = lambda vk: st.fmean([a - b for a, b in zip(val(vk, "dabs20"), val("full", "dabs20"))])


def best(vals, fmt):
    """Bold the smallest entry (ties at the printed precision are all bold)."""
    txt = [fmt.format(v) for v in vals]
    lo = min(txt, key=float)
    return [f"\\textbf{{{t}}}" if t == lo else t for t in txt]


NICE = {"fog": "Fog", "frost": "Frost", "snow": "Snow", "motion_blur": "Motion blur", "spatter": "Spatter", "zoom_blur": "Zoom blur", "elastic_transform": "Elastic",
        "speckle_noise": "Speckle", "gaussian_noise": "Gaussian noise", "impulse_noise": "Impulse noise", "shot_noise": "Shot noise", "jpeg_compression": "JPEG",
        "glass_blur": "Glass blur", "defocus_blur": "Defocus blur", "rain": "Rain", "gaussian_blur": "Gaussian blur", "pixelate": "Pixelate", "contrast": "Contrast",
        "saturate": "Saturate", "brightness": "Brightness"}
STAGED = [("e0", "Clean-only baseline"), ("x4", "Clean--corrupted consistency"), ("no_mono", "+ corr.\\ dropout + overlays"), ("full", "Full model")]
T = []


def begin(title, spec, head):
    return [f"% ---------- {title}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule", head + " \\\\", "\\midrule"]


END = ["\\bottomrule", "\\end{tabular}", ""]

# Table 1: staged variants, three-run means
cols = ["clean_abs", "clean_d1", "dabs20", "fog", "frost", "snow", "motion_blur"]
fm = {"clean_abs": "{:.3f}", "clean_d1": "{:.3f}", "dabs20": "{:.3f}", "fog": "{:.2f}", "frost": "{:.2f}", "snow": "{:.2f}", "motion_blur": "{:.2f}"}
cells = {c: best([mean(vk, c) for vk, _ in STAGED], fm[c]) for c in cols}
T += begin("Table 1: primary 28-frame local results, three training runs per method", "lccccccc", "Method & Abs & D1 & $\\Delta$Abs & Fog & Frost & Snow & M.\\ blur")
for i, (vk, lab) in enumerate(STAGED):
    d = f"$\\mathbf{{{mean(vk, 'dabs20'):.3f}\\pm{sd(vk, 'dabs20'):.3f}}}$" if "textbf" in cells["dabs20"][i] else pm(vk)
    T.append(f"{lab} & {cells['clean_abs'][i]} & {cells['clean_d1'][i]} & {d} & " + " & ".join(cells[c][i] for c in cols[3:]) + " \\\\")
T += END

# Table 2: ablations
ABL = [("full", "Full model"), ("no_dropout", "$-$ correspondence dropout"), ("randkey", "Random-key dropout"), ("no_delta", "$-\\,\\mathcal{L}_\\Delta$"),
       ("image_plane", "Image-plane mask transport"), ("no_teacher", "$-$ teacher / anchor")]
T += begin("Table 2: ablations, three seeds; last column = mean matched-seed difference to the full model", "lcccc", "Variant & Seeds & Clean Abs & $\\Delta$Abs (20) & $\\Delta$ vs.\\ Full")
T += [f"{lab} & 3 & {mean(vk, 'clean_abs'):.3f} & {pm(vk)} & " + ("--" if vk == "full" else f"$+{dmean(vk):.3f}$") + " \\\\" for vk, lab in ABL] + END

# Table 5: clean-only baseline per corruption
order5 = sorted(["fog", "frost", "snow", "motion_blur", "spatter", "zoom_blur", "elastic_transform"], key=lambda c: -mean("e0", c))
T += begin("Table 5: clean-only baseline, mean over three training runs", "lc@{\\qquad}lc", "Corruption & $\\Delta$Abs $\\downarrow$ & Corruption & $\\Delta$Abs $\\downarrow$")
for a, b in zip(order5[:4], order5[4:] + [None]):
    T.append(f"{NICE[a]} & {mean('e0', a):.2f} & " + (f"{NICE[b]} & {mean('e0', b):.2f}" if b else f"Mean (20) & {mean('e0', 'dabs20'):.3f}") + " \\\\")
T += END

# Table 6: individual runs
T6 = [("full", "Full"), ("no_dropout", "$-$ correspondence dropout"), ("randkey", "Random-key dropout"), ("no_delta", "$-\\,\\mathcal{L}_\\Delta$"),
      ("no_mono", "$-$ monocular branch"), ("no_overlay", "$-$ sensor overlays"), ("image_plane", "Image-plane mask transport"), ("no_teacher", "$-$ teacher / anchor")]
T += begin("Table 6: individual runs (same position = same seed)", "lcc", "Variant & Individual runs & Mean $\\pm$ sd")
T += [f"{lab} & {runs(vk)} & {pm(vk)} \\\\" for vk, lab in T6] + END

# Table 7: sky-region drift
rm = lambda key: st.fmean([R[str(s)][key] for s in K.SEEDS])
T += begin("Table 7: mean sky-region drift (px), three-run means; reduction from the unrounded means", "lccc", "Corruption & Clean-only & Full & Reduction")
for c in ("fog", "frost", "snow"):
    a, b = rm(f"e0|{c}|sky"), rm(f"full|{c}|sky")
    T.append(f"{NICE[c]} & {a:.1f} & {b:.1f} & {100 * (1 - b / a):.0f}\\% \\\\")
T += END


def percorr(title, corrs, mean_row):
    out = begin(title, "lcccc", "Corruption & Clean-only & Consistency & + Corr.\\ dropout + overlays & Full")
    out += [f"{NICE[c]} & " + " & ".join(best([mean(vk, c) for vk, _ in STAGED], "{:.2f}")) + " \\\\" for c in corrs]
    if mean_row:
        out += ["\\midrule", "Mean & " + " & ".join(best([mean(vk, "dabs20") for vk, _ in STAGED], "{:.3f}")) + " \\\\"]
    return out + END


T += percorr("Table 8: held-out corruptions, three-run means", ["fog", "frost", "speckle_noise", "zoom_blur", "elastic_transform"], False)

# Table 10: additional variants
OTH = [("x2", "Short robust fine-tuning"), ("x6a", "Reliability gating"), ("x5b", "Attention supervision"), ("x9", "Attention supervision + corr.\\ dropout")]
T += begin("Table 10: additional variants, three training runs each", "lcccc", "Variant & Clean Abs & Clean D1 & Individual runs & $\\Delta$Abs (20)")
T += [f"{lab} & {mean(vk, 'clean_abs'):.3f} & {mean(vk, 'clean_d1'):.3f} & {runs(vk)} & {pm(vk)} \\\\" for vk, lab in OTH] + END

# Table 11: CorrMask probability
frac = {vk: st.fmean([K.train_stats(K.run(vk, s)["train"])["corrmask_event_frac"] for s in K.SEEDS]) for vk in ("cm_p000", "cm_p010", "full", "cm_p050")}
T += begin("Table 11: CorrMask sampling probability; masked samples = observed fraction of all training samples", "lcccc",
           "$p_{\\mathrm{mask}}$ & Masked samples & Clean Abs & Individual runs & $\\Delta$Abs (20)")
for vk, lab in (("cm_p000", "0 (no CorrMask)"), ("cm_p010", "0.10"), ("full", "0.25 (default)"), ("cm_p050", "0.50")):
    T.append(f"{lab} & {100 * frac[vk]:.1f}\\% & {mean(vk, 'clean_abs'):.3f} & {runs(vk)} & {pm(vk)} \\\\")
T += END

# Table 12: loss weights
LW = [("no_delta", "$\\lambda_\\Delta=0$"), ("ld050", "$\\lambda_\\Delta=0.5$"), ("full", "$\\lambda_\\Delta=1$ (default)"), ("ld200", "$\\lambda_\\Delta=2$"),
      ("lc0125", "$\\lambda_{\\mathrm{corr}}=0.125$"), ("lc050", "$\\lambda_{\\mathrm{corr}}=0.5$")]
ld = sorted(val("ld200", "clean_abs"))
T += begin(f"Table 12: one-factor loss-weight sensitivity; dagger: one lambda_delta = 2 run ended with clean Abs {ld[2]:.3f}, the other two {ld[0]:.3f} and {ld[1]:.3f}",
           "lcccc", "Setting & Clean Abs & Clean D1 & Individual runs & $\\Delta$Abs (20)")
for vk, lab in LW:
    star = "$^{\\dagger}$" if vk == "ld200" else ""
    T.append(f"{lab} & {mean(vk, 'clean_abs'):.3f}{star} & {mean(vk, 'clean_d1'):.3f}{star} & {runs(vk)} & {pm(vk)} \\\\")
T += END

T += percorr("Table 13: all corruptions, three-run means", sorted(K.CORRUPTIONS20, key=lambda c: -mean("e0", c)), True)

# Table 15: all 49 validation frames
c15 = best([mean(vk, "all49_dabs20") for vk, _ in STAGED], "{:.3f}")
T += begin("Table 15: all 49 local validation frames, three-run means", "lcc", "Method & Clean Abs & $\\Delta$Abs (20)")
T += [f"{lab.replace('+ corr.', '+ Corr.')} & {mean(vk, 'all49_clean_abs'):.3f} & {c15[i]} \\\\" for i, (vk, lab) in enumerate(STAGED)] + END

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text("\n".join(T))
print("\n".join(T))

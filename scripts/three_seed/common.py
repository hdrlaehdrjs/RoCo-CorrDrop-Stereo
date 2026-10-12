"""Registry and readers of the three-seed local experiments (seeds 2026 / 2027 / 2028).

runs.json lists, for every variant and seed, the config name and the training / evaluation folders of our runs
(relative to $ROCO_TRACK2_WORK/outputs/track2). A run retrained with

    scripts/run_train_eval.sh <gpu> <config> <config>

is found automatically in outputs/track2/final_runs/<config>/ and takes precedence over the recorded folder.

All numbers are local development results (scene-disjoint split, 28 primary validation frames, frozen robust20 cache),
not official RobustSpring scores."""
from __future__ import annotations

import hashlib
import json
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
WORK = Path(os.environ.get("ROCO_TRACK2_WORK", ROOT))
OUTPUTS = WORK / "outputs/track2"
R20 = OUTPUTS / "robust20_val"
OUT = OUTPUTS / "three_seed"                     # default output folder of the scripts in this directory
SEEDS = (2026, 2027, 2028)
PRIMARY = ("0038", "0039", "0045", "0047")
N_PRIMARY, N_VAL = 28, 49
CORRUPTIONS20 = ("brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                 "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
                 "rain", "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur")
LOSS_KEYS = ("l_clean", "l_corr", "l_delta", "l_anchor", "l_mono_clean", "l_mono_corr")
IGNORE = {"name", "config_file", "seed"}

_REG = json.loads((HERE / "runs.json").read_text())
VARIANTS = _REG["variants"]
REPETITIONS, ABORTED = _REG["repetitions"], _REG["aborted"]


def locate(entry: dict) -> tuple[Path, Path]:
    """(training folder, evaluation folder) of one registry entry."""
    own = OUTPUTS / "final_runs" / entry["config"]
    if (own / "train/contract.json").exists():
        return own / "train", own / "eval/robust20"
    return OUTPUTS / entry["train"], OUTPUTS / entry["eval"]


def run(variant: str, seed: int) -> dict:
    e = VARIANTS[variant]["seeds"][str(seed)]
    train, ev = locate(e)
    return dict(variant=variant, seed=seed, config=e["config"], train=train, eval=ev)


def rel(p) -> str:
    p = Path(p)
    return str(p.relative_to(OUTPUTS)) if OUTPUTS in p.parents else str(p)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def flat(d: dict, p: str = "") -> dict:
    o = {}
    for k, v in d.items():
        o.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return o


def val_keys() -> list[str]:
    return (ROOT / "splits/track2_val.txt").read_text().split()


def load_eval(eval_dir: Path) -> tuple[dict | None, str]:
    """Strict reader of one robust20 evaluation (rows.jsonl of scripts/eval_suite.py).

    Requires all 49 validation frames x (clean + 20 corruptions) = 1029 unique rows, 588 of them on the 28 primary
    frames. Metrics on the primary frames:
      clean_abs / clean_d1 : mean over the 28 clean rows
      dabs20               : unweighted mean over the 20 corruptions of the per-corruption mean of the per-frame
                             'delta' (mean |d_corr - d_clean| over all pixels)
    Returns (metrics, 'ok') or (None, reason)."""
    p = eval_dir / "rows.jsonl"
    if not p.exists():
        return None, "rows.jsonl missing"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    keys = [(r["frame"], r["condition"]) for r in rows]
    if len(set(keys)) != len(keys):
        return None, f"duplicate frame-condition rows ({len(keys) - len(set(keys))})"
    want_frames, want_conds = set(val_keys()), set(CORRUPTIONS20) | {"clean"}
    if {k[0] for k in keys} != want_frames or {k[1] for k in keys} != want_conds or len(keys) != N_VAL * 21:
        return None, f"incomplete: {len(keys)} rows, {len({k[0] for k in keys})} frames, {len({k[1] for k in keys})} conditions (need 1029 / 49 / 21)"
    prim = [r for r in rows if r["frame"].split(":")[0] in PRIMARY]
    per: dict[str, list] = {}
    for r in prim:
        per.setdefault(r["condition"], []).append(r)
    if len(prim) != N_PRIMARY * 21 or any(len(v) != N_PRIMARY for v in per.values()):
        return None, f"primary subset incomplete: {len(prim)} rows"
    m = lambda c, k: st.fmean(x[k] for x in per[c])
    pc = {c: m(c, "delta") for c in CORRUPTIONS20}
    res = dict(clean_abs=m("clean", "abs"), clean_d1=m("clean", "d1"), clean_1px=m("clean", "px1"),
               dabs20=st.fmean(pc.values()), per_corruption=pc,
               all49_clean_abs=st.fmean(r["abs"] for r in rows if r["condition"] == "clean"),
               all49_dabs20=st.fmean(st.fmean(r["delta"] for r in rows if r["condition"] == c) for c in CORRUPTIONS20))
    return res, "ok"


def train_stats(train_dir: Path) -> dict:
    """Facts of one training run from its own contract.json / status.json / train.jsonl."""
    out = {}
    cj, sj, tj = train_dir / "contract.json", train_dir / "status.json", train_dir / "train.jsonl"
    if cj.exists():
        c = json.loads(cj.read_text())
        out.update(seed=c.get("seed"), n_train=c.get("split", {}).get("n_train"), teacher_used=c.get("teacher_used"),
                   init_sha=c.get("init_checkpoint_sha256"), teacher_sha=c.get("teacher_checkpoint_sha256"),
                   config=c.get("config"), gpu=c.get("gpu"))
    if sj.exists():
        s = json.loads(sj.read_text())
        out.update(status=s.get("phase"), steps_done=s.get("steps"), wall_hours=s.get("wall_seconds", 0) / 3600)
    if tj.exists():
        rows = [json.loads(l) for l in tj.read_text().splitlines() if l.strip()]
        n = sum(len(r["aug"]) for r in rows)
        out.update(steps_logged=len(rows), samples=n,
                   aug_frac=sum(a != "clean" for r in rows for a in r["aug"]) / max(1, n),
                   corrmask_event_frac=sum(r.get("mask_event", 0) for r in rows) / max(1, n),
                   peak_gib=max((r["peak_gib"] for r in rows if "peak_gib" in r), default=None))
    return out


def nonfinite_steps(train_dir: Path, ref_dir: Path) -> int:
    """Steps with a non-finite loss term or gradient norm. Terms that are NaN in every step of the seed-2026 run of the
    same variant are placeholders of the recipe (the clean-only baseline has no corrupted branch and logs
    l_corr = NaN) and are not counted."""
    keys = LOSS_KEYS + ("grad_norm",)
    read = lambda d: [json.loads(l) for l in (d / "train.jsonl").read_text().splitlines() if l.strip()]
    ref = read(ref_dir)
    by_design = {k for k in keys if all(k in x and x[k] != x[k] for x in ref)}
    return sum(1 for x in read(train_dir) if any(k in x and x[k] != x[k] for k in keys if k not in by_design))


if __name__ == "__main__":      # the 63 config names, the seed-2026 clean-only baseline (teacher of the robust runs) first
    names = [e["config"] for v in VARIANTS.values() for e in v["seeds"].values()]
    first = VARIANTS["e0"]["seeds"]["2026"]["config"]
    print("\n".join([first] + [n for n in names if n != first]))

#!/usr/bin/env python3
"""Consistency of the 63 configurations behind the three-seed tables (CPU only, no run needed).

  python3 scripts/three_seed/check_configs.py            # config files only
  python3 scripts/three_seed/check_configs.py --runs     # additionally compare with the contract of every available run

Checks, per variant:
  - every config resolves, carries the registered seed and uses the 4,054-pair local training split;
  - the configs of seeds 2027 and 2028 equal the seed-2026 config except for seed and name;
  - a sensitivity setting differs from the full model in exactly its one coefficient.
With --runs the configuration recorded in contract.json of each run must equal the resolved config file."""
import argparse
import json

import common as K
from track2.config import load_config

FULL = "track2_croco_final_base"


def diff(a: dict, b: dict, ignore=K.IGNORE) -> dict:
    a, b = K.flat(a), K.flat(b)
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k) and k not in ignore}


def check(with_runs: bool = False) -> tuple[list[str], list[str]]:
    """Returns (report lines, problems)."""
    full = load_config(FULL)
    lines = ["| variant | seed | config | difference to the seed-2026 config | difference to the full model | ok |", "|---|---|---|---|---|---|"]
    problems = []
    for vk, v in K.VARIANTS.items():
        ref = load_config(v["seeds"]["2026"]["config"])
        for seed in K.SEEDS:
            e = v["seeds"][str(seed)]
            cfg = load_config(e["config"])
            p = []
            if cfg.get("seed", 2026) != seed:
                p.append(f"seed {cfg.get('seed')}")
            if cfg.get("train_split", "train") != "train":
                p.append("not the local training split")
            d_ref = diff(ref, cfg)
            if d_ref:
                p.append("differs from the seed-2026 config")
            d_full = diff(full, cfg)
            if "factor" in v:
                key, val = v["factor"]
                if set(d_full) != {key} or d_full[key][1] != val:
                    p.append(f"expected only {key} = {val} to differ from the full model")
            if with_runs:
                train, _ = K.locate(e)
                if (train / "contract.json").exists():
                    rec = json.loads((train / "contract.json").read_text())
                    if diff(rec["config"], cfg, ignore={"config_file"}) or rec["seed"] != seed:
                        p.append("recorded run configuration differs from the config file")
                else:
                    p.append("run not found")
            shown = "; ".join(f"`{k}`: {a} → {b}" for k, (a, b) in d_full.items()) if v["one_factor_ablation_of_full"] and vk != "full" else "(different recipe)" if vk != "full" else "–"
            lines.append(f"| {v['label']} | {seed} | `{e['config']}` | {d_ref or 'none'} | {shown} | {'yes' if not p else '**NO**: ' + '; '.join(p)} |")
            problems += [f"{vk} seed {seed}: {x}" for x in p]
    return lines, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="store_true", help="also compare with the contract.json of every available run")
    ap.add_argument("--out", help="write the report to this markdown file")
    a = ap.parse_args()
    lines, problems = check(a.runs)
    n = sum(len(v["seeds"]) for v in K.VARIANTS.values())
    text = "\n".join(["# Configuration check of the three-seed experiments", ""] + lines + ["", f"{n} configurations checked, {len(problems)} problems."]) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()

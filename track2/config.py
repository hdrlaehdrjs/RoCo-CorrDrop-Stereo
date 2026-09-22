"""YAML configs with single inheritance (`inherit: other_config_name`) and deep merge."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .paths import ROOT, WORK

CONFIG_DIR = ROOT / "configs/track2"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(name_or_path: str, overrides: list[str] | None = None) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    raw = yaml.safe_load(p.read_text()) or {}
    parent = raw.pop("inherit", None)
    cfg = _merge(load_config(parent), raw) if parent else raw
    for o in overrides or []:
        key, val = o.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)
    for k in ("init_checkpoint", "teacher_checkpoint"):      # relative checkpoint paths live in the working directory
        if cfg.get(k) and not Path(cfg[k]).is_absolute():
            cfg[k] = str(WORK / cfg[k])
    cfg["config_file"] = str(p)
    return cfg

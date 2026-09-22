#!/usr/bin/env python3
"""Run the fixed Track 2 validation suite for one stereo model.

Backbones run in separate processes (CroCo and DEFOM share top-level module names such
as `utils`). Rows are appended to rows.jsonl so an interrupted run resumes exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__import__("os").environ.get("ROCO_TRACK2_WORK", ROOT))   # checkpoints/, outputs/, third_party/croco live here
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track2 import val_suite as vs  # noqa: E402
from track2.paths import TRACK3_ROOT  # noqa: E402
from track2.spring_data import load_split, read_frame  # noqa: E402


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


class CroCoAdapter:
    def __init__(self, ckpt, overlap, amp, tile_batch):
        from track2.croco_model import build_croco_stereo
        self.model, self.meta = build_croco_stereo(ckpt, "cuda")
        self.model.eval()
        self.overlap, self.tile_batch = overlap, tile_batch
        self.amp = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[amp]
        self.desc = dict(backbone="CroCo-Stereo ViT-L/Base", checkpoint=str(ckpt), tile_overlap=overlap,
                         tile_crop=self.meta["crop"], tile_conf_mode=self.meta["tile_conf_mode"], amp=amp,
                         resolution="native 1080p, tiled")

    def __call__(self, left, right):
        l = torch.from_numpy(left).permute(2, 0, 1)[None].float().cuda()
        r = torch.from_numpy(right).permute(2, 0, 1)[None].float().cuda()
        return self.model.tiled_predict(l, r, overlap=self.overlap, tile_batch=self.tile_batch, amp_dtype=self.amp)[0, 0]


class DEFOMAdapter:
    """Uses the validated Track 3 DEFOM inference path unchanged (read-only import)."""

    def __init__(self, ckpt, iters, scale_iters, state_key=None):
        sys.path[:0] = [str(TRACK3_ROOT / "scripts")]
        import track3_defom_dpflow as base
        self.base = base
        if state_key:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)[state_key]
            tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"defom_{os.getpid()}.pth"
            torch.save(state, tmp)
            self.model = base.load_stereo(tmp)
            tmp.unlink()
        else:
            self.model = base.load_stereo(Path(ckpt))
        self.iters, self.scale_iters = iters, scale_iters
        self.desc = dict(backbone="DEFOM-Stereo ViT-S", checkpoint=str(ckpt), state_key=state_key, iters=iters,
                         scale_iters=scale_iters, resolution="native 1080p, pad /32", amp="none")

    def __call__(self, left, right):
        l = torch.from_numpy(left).permute(2, 0, 1)[None].float().cuda()
        r = torch.from_numpy(right).permute(2, 0, 1)[None].float().cuda()
        return self.base.infer_stereo(self.model, l, r, self.iters, self.scale_iters)


class FoundationStereoAdapter:
    """Official FoundationStereo inference (third_party/FoundationStereo scripts/run_demo.py): RGB 0..255,
    InputPadder /32, autocast fp16, hierarchical inference for >1K images (small_ratio 0.5), valid_iters 32."""

    def __init__(self, ckpt, iters=32, hiera=1):
        root = WORK / "third_party/FoundationStereo"
        sys.path[:0] = [str(root)]
        import types
        for mod in ("trimesh", "open3d", "joblib"):  # imported by Utils.py but unused on the inference path (visualisation only)
            if mod not in sys.modules:
                try:
                    __import__(mod)
                except ImportError:
                    sys.modules[mod] = types.ModuleType(mod)
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        cfg = OmegaConf.load(str(Path(ckpt).parent / "cfg.yaml"))
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"], cfg["hiera"] = iters, hiera
        self.model = FoundationStereo(cfg)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        self.model.cuda().eval()
        self.Padder, self.iters, self.hiera = InputPadder, iters, hiera
        self.desc = dict(backbone=f"FoundationStereo {cfg['vit_size']} (zero-shot)", checkpoint=str(ckpt), valid_iters=iters,
                         hiera=hiera, amp="autocast fp16 (official)", resolution="native 1080p")

    @torch.no_grad()
    def __call__(self, left, right):
        l = torch.from_numpy(left).permute(2, 0, 1)[None].float().cuda()
        r = torch.from_numpy(right).permute(2, 0, 1)[None].float().cuda()
        pad = self.Padder(l.shape, divis_by=32, force_square=False)
        l, r = pad.pad(l, r)
        with torch.cuda.amp.autocast(True):
            d = self.model.run_hierachical(l, r, iters=self.iters, test_mode=True, small_ratio=0.5) if self.hiera else \
                self.model.forward(l, r, iters=self.iters, test_mode=True)
        return pad.unpad(d.float())[0, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", choices=["croco", "defom", "foundationstereo"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", type=Path, default=WORK / "outputs/track2/eval")
    p.add_argument("--overlap", type=float, default=0.7)
    p.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    p.add_argument("--tile-batch", type=int, default=1)  # 1 = bit-identical to official tiled_pred
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--scale-iters", type=int, default=6)
    p.add_argument("--state-key", default=None)
    p.add_argument("--frames", type=int, default=0, help="limit frames (smoke only)")
    p.add_argument("--conditions", default="all")
    p.add_argument("--suite", default="local9", choices=["local9", "robust20"],
                   help="local9: on-the-fly 9-corruption suite; robust20: frozen RobustSpring-style 20-corruption images")
    p.add_argument("--split", default="val", help="val (fixed suite) or trainprobe (reproduction check on train scenes)")
    a = p.parse_args()

    out = a.out / a.tag
    out.mkdir(parents=True, exist_ok=True)
    frames = load_split(a.split)
    if a.frames:
        frames = frames[:a.frames]
    if a.suite == "robust20":
        from track2 import robust20 as R
        from PIL import Image
        R20 = WORK / "outputs/track2/robust20_val"
        manifest = json.loads((R20 / "manifest.json").read_text())
        assert a.split == "val" and manifest["frames"] == [f.key for f in load_split("val")]
        ALL = list(R.CORRUPTIONS20)
    else:
        ALL = list(vs.CORRUPTIONS)
    conds = ALL if a.conditions == "all" else ([] if a.conditions == "clean" else a.conditions.split(","))
    torch.backends.cudnn.benchmark = False
    if a.backbone == "croco":
        model = CroCoAdapter(a.checkpoint, a.overlap, a.amp, a.tile_batch)
    elif a.backbone == "foundationstereo":
        model = FoundationStereoAdapter(a.checkpoint)
    else:
        model = DEFOMAdapter(a.checkpoint, a.iters, a.scale_iters, a.state_key)
    contract = dict(model=model.desc, frames=[f.key for f in frames], conditions=["clean"] + conds,
                    suite=dict(severity=vs.SEVERITY, base_seed=vs.BASE_SEED, metric_doc=vs.__doc__) if a.suite == "local9" else
                    dict(name="robust20", manifest_sha256=sha256(R20 / "manifest.json"), severity=manifest["severity"]),
                    checkpoint_sha256=sha256(a.checkpoint), torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), host=platform.node(), argv=sys.argv,
                    visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    cpath = out / "contract.json"
    if cpath.exists():
        old = json.loads(cpath.read_text())
        for k in ("model", "frames", "conditions", "checkpoint_sha256", "suite"):
            if old[k] != contract[k]:
                raise RuntimeError(f"contract mismatch on {k}; use a new tag")
    else:
        cpath.write_text(json.dumps(contract, indent=2))
    rows_path = out / "rows.jsonl"
    done = set()
    rows = []
    if rows_path.exists():
        rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
        done = {(r["frame"], r["condition"]) for r in rows}
    cond_index = {c: i for i, c in enumerate(vs.CORRUPTIONS)}
    val_keys = [f.key for f in load_split(a.split)]
    t0 = time.time()
    for fi, fr in enumerate(frames):
        need = [c for c in ["clean"] + conds if (fr.key, c) not in done]
        if not need:
            continue
        item = read_frame(fr, full_gt=True)
        gfull = torch.from_numpy(item["disp_full"]).cuda()
        gstr = torch.from_numpy(item["disp"]).cuda()
        tick = time.time()
        clean = model(item["left"], item["right"]).float()
        assert torch.isfinite(clean).all()
        new = []
        if "clean" in need:
            new.append(dict(frame=fr.key, condition="clean", **vs.frame_metrics(clean, gfull, gstr)))
        for c in need:
            if c == "clean":
                continue
            fi_global = val_keys.index(fr.key)
            if a.suite == "robust20":
                load = lambda side: np.asarray(Image.open(R20 / c / fr.scene / f"frame_{side}_{fr.frame:04d}.png").convert("RGB")).copy()
                Lc, Rc = load("left"), load("right")
            else:
                Lc, Rc = vs.corrupt_pair(item["left"], item["right"], c, vs.condition_seed(cond_index[c], fi_global))
            pred = model(Lc, Rc).float()
            assert torch.isfinite(pred).all()
            m = vs.frame_metrics(pred, gfull, gstr)
            valid = torch.isfinite(gstr)
            m.update(delta=float((pred - clean).abs().mean()), delta_valid=float((pred - clean)[valid].abs().mean()))
            new.append(dict(frame=fr.key, condition=c, **m))
        with rows_path.open("a") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
        rows += new
        print(f"[{a.tag}] frame {fi+1}/{len(frames)} {fr.key} {len(need)} preds {time.time()-tick:.1f}s "
              f"clean_abs={[r['abs'] for r in rows if r['frame']==fr.key and r['condition']=='clean']}", flush=True)
    s = vs.summarize(rows, corruptions=ALL, groups=None if a.suite == "local9" else R.GROUPS20)
    s["suite"] = a.suite
    s.update(tag=a.tag, model=model.desc, frames=len(frames), wall_seconds=time.time() - t0,
             peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print("SUMMARY", json.dumps({k: s.get(k) for k in ("Clean_Abs", "Clean_D1", "DeltaAbs_local", "LOCAL_PROXY_RBS")}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Spring / RobustSpring stereo submission for one CroCo checkpoint (Track 2).

Official format (roco-spring-devkit README + spring-benchmark.org submission page, checked 2026-09-18):
  raw predictions   <root>/<condition>/test/<scene>/disp1_{left,right}/disp1_{left,right}_XXXX.dsp5
                    condition in {clean} + 20 RobustSpring corruptions; dsp5 = HDF5 dataset "disparity",
                    float32 H x W (1080 x 1920), gzip (flow_IO.writeDsp5File). The official disp1 tools read BOTH
                    views (1000 frames x 2 = 2000 files per condition; verified on the Track 3 subsampling logs).
                    Right-view disparity = the same network on the horizontally flipped, swapped pair, flipped back.
  upload files      ONLY the two HDF5 files produced by the official subsampling binaries:
                    disp1_subsampling <root>/clean/test        -> disp1_submission.hdf5   (clean entry)
                    disp1_robust_subsampling <root>            -> disp1_robustness.hdf5   (robust entry, 21 groups)
  Track 3 lesson: never hand-pack predictions into an HDF5 (a 22 GiB bundle was produced that way). The bundle
  step refuses anything that does not look like the official output (size / shape / dtype / group checks).

Sub-commands
  predict --checkpoint CK --root ROOT [--conditions all|c1,c2] [--worker i/n]    resumable, one process per GPU
  check   --root ROOT                                                              counts / shapes / finiteness
  bundle  --root ROOT --out BUNDLE                                                 official binaries + verification
The same checkpoint and the same inference settings (fp32, tiles 352x704, overlap 0.7, tile_batch 1 = official
tiled_pred) are used for clean and every corruption, as the challenge rules require.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(__import__("os").environ.get("ROCO_DATA", "/DLMATH/Data/roco"))
TOOLS = Path(__import__("os").environ.get("ROCO_SUBSAMPLING_TOOLS", "/DLMATH/KDG/Image/Roco/tools/spring_subsampling/subsampling_tools"))
CORRUPTIONS = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise",
               "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "rain", "saturate", "shot_noise",
               "snow", "spatter", "speckle_noise", "zoom_blur"]
CONDITIONS = ["clean"] + CORRUPTIONS
H, W = 1080, 1920
N_TEST = 1000                     # test frames (10 scenes); x 2 views = 2000 dsp5 files per condition
VIEWS = ("left", "right")
CLEAN_SHAPE = (124414000,)        # official disp1_submission.hdf5 'disparity' (Track 3 reference bundle)
ROBUST_SHAPE = (5183900,)         # official disp1_robustness.hdf5 per-condition 'disparity'
MAX_BUNDLE_BYTES = 1 << 30        # anything above 1 GiB is not an official subsampled file


def test_frames():
    out = []
    for scene in sorted(p.name for p in (DATA / "spring/test").iterdir()):
        for png in sorted((DATA / "spring/test" / scene / "frame_left").glob("frame_left_*.png")):
            out.append((scene, int(png.stem.split("_")[-1])))
    assert len(out) == N_TEST, len(out)
    return out


def image_dir(condition, scene):
    return DATA / "spring/test" / scene if condition == "clean" else DATA / "robust_spring" / condition / "test" / scene


def dsp5_path(root, condition, scene, frame, view="left"):
    return root / condition / "test" / scene / f"disp1_{view}" / f"disp1_{view}_{frame:04d}.dsp5"


def predict_pair(model, L, R, overlap, view):
    """Left disparity of (L, R); the right view is predicted on the flipped, swapped pair and flipped back."""
    import torch
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().cuda()
    with torch.no_grad():
        if view == "left":
            out = model.tiled_predict(to(L), to(R), overlap=overlap, tile_batch=1)[0, 0]
        else:
            out = model.tiled_predict(to(R[:, ::-1]), to(L[:, ::-1]), overlap=overlap, tile_batch=1)[0, 0].flip(-1)
    return np.ascontiguousarray(out.float().cpu().numpy(), dtype=np.float32)


def write_dsp5(disp: np.ndarray, path: Path):
    """flow_IO.writeDsp5File, written atomically."""
    assert disp.dtype == np.float32 and disp.shape == (H, W) and np.isfinite(disp).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".dsp5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("disparity", data=disp, compression="gzip", compression_opts=5)
    os.replace(tmp, path)


def read_dsp5(path: Path):
    with h5py.File(path, "r") as f:
        return f["disparity"][()]


def cmd_predict(a):
    import torch
    from PIL import Image
    from track2.croco_model import build_croco_stereo
    conds = CONDITIONS if a.conditions == "all" else a.conditions.split(",")
    assert all(c in CONDITIONS for c in conds), conds
    wi, wn = (int(x) for x in a.worker.split("/"))
    jobs = [(c, s, f, v) for c in conds for (s, f) in test_frames() for v in VIEWS]
    # dynamic sharing: every worker walks the WHOLE job list from its own offset and skips files that exist or that
    # another worker is writing (fresh .lock), so fast GPUs automatically take over work from slow / shared GPUs
    start = (wi * len(jobs)) // wn
    jobs = jobs[start:] + jobs[:start]
    root = Path(a.root)
    remaining = sum(1 for j in jobs if not dsp5_path(root, *j).exists())
    print(f"[worker {wi}/{wn}] {len(jobs)} jobs in total, {remaining} still missing (shared between workers)", flush=True)
    if not remaining:
        return
    model, meta = build_croco_stereo(a.checkpoint, "cuda")
    model.eval()
    torch.backends.cudnn.benchmark = False
    desc = dict(checkpoint=a.checkpoint, checkpoint_sha256=sha256(a.checkpoint), tile_crop=meta["crop"], overlap=a.overlap,
                tile_batch=1, amp="none (fp32)", tile_conf_mode=meta["tile_conf_mode"], care=meta.get("care"), mono=meta.get("mono"))
    (root / "inference_contract.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "inference_contract.json").write_text(json.dumps(desc, indent=2))
    t0 = time.time()
    done = 0
    for k, (c, s, f, v) in enumerate(jobs):
        target = dsp5_path(root, c, s, f, v)
        lock = target.with_suffix(".lock")
        if target.exists():
            continue
        if not a.sweep and lock.exists() and time.time() - lock.stat().st_mtime < 600:
            continue
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.touch()
        d = image_dir(c, s)
        L = np.asarray(Image.open(d / "frame_left" / f"frame_left_{f:04d}.png").convert("RGB"))
        R = np.asarray(Image.open(d / "frame_right" / f"frame_right_{f:04d}.png").convert("RGB"))
        assert L.shape == (H, W, 3) and R.shape == (H, W, 3), (c, s, f, L.shape)
        write_dsp5(predict_pair(model, L, R, a.overlap, v), target)
        lock.unlink(missing_ok=True)
        done += 1
        if done % 20 == 1:
            el = time.time() - t0
            print(f"[worker {wi}/{wn}] wrote {done} (list position {k + 1}/{len(jobs)}) {c}/{s}/{f:04d}/{v} {el / done:.1f}s/pred", flush=True)
    print(f"[worker {wi}/{wn}] finished: wrote {done} files in {(time.time() - t0) / 3600:.2f} h", flush=True)


def cmd_check(a):
    root = Path(a.root)
    conds = CONDITIONS if a.conditions == "all" else a.conditions.split(",")
    frames = test_frames()
    report = {}
    ok = True
    n_files = N_TEST * len(VIEWS)
    for c in conds:
        missing, bad = [], []
        for s, f in frames:
            for v in VIEWS:
                p = dsp5_path(root, c, s, f, v)
                if not p.exists():
                    missing.append(f"{s}/{f:04d}/{v}")
                    continue
                if a.deep:
                    try:
                        d = read_dsp5(p)
                        if d.shape != (H, W) or d.dtype != np.float32 or not np.isfinite(d).all():
                            bad.append(f"{s}/{f:04d}/{v}:{d.shape}:{d.dtype}")
                    except Exception as e:  # noqa: BLE001
                        bad.append(f"{s}/{f:04d}/{v}:{e}")
        report[c] = dict(present=n_files - len(missing), missing=len(missing), bad=len(bad), examples=(missing + bad)[:5])
        ok &= not missing and not bad
        print(f"{c:18s} present {n_files - len(missing):4d}/{n_files} bad {len(bad)}", flush=True)
    (root / "check_report.json").write_text(json.dumps(report, indent=2))
    print("CHECK", "OK" if ok else "INCOMPLETE")
    sys.exit(0 if ok else 1)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cmd_bundle(a):
    root = Path(a.root).resolve()
    out = Path(a.out).resolve()
    for c in CONDITIONS:
        n = sum(1 for s, f in test_frames() for v in VIEWS if dsp5_path(root, c, s, f, v).exists())
        assert n == N_TEST * len(VIEWS), f"{c}: {n}/{N_TEST * len(VIEWS)} predictions; run `check` first"
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "robust").mkdir(parents=True, exist_ok=True)
    runs = [("clean", TOOLS / "disp1_subsampling", root / "clean" / "test", "disp1_submission.hdf5"),
            ("robust", TOOLS / "disp1_robust_subsampling", root, "disp1_robustness.hdf5")]
    for sub, exe, arg, expect in runs:
        assert exe.is_file(), exe
        target = out / sub / expect
        if target.exists() and not a.force:
            print(f"{target} exists; skipping (use --force to redo)")
            continue
        for old in (out / sub).glob("*.hdf5"):
            old.unlink()
        log = out / sub / f"{sub}_subsampling.log"
        with log.open("w") as lf:
            rc = subprocess.call([str(exe), str(arg)], cwd=out / sub, stdout=lf, stderr=subprocess.STDOUT)
        produced = sorted((out / sub).glob("*.hdf5"))
        assert rc == 0 and len(produced) == 1, f"{exe.name} rc={rc} produced={produced}; see {log}"
        if produced[0].name != expect:
            produced[0].rename(target)
    # ---- verification against the official layout (the 22 GiB mistake is impossible to repeat past this point)
    summary = {}
    for sub, expect, shape in (("clean", "disp1_submission.hdf5", CLEAN_SHAPE), ("robust", "disp1_robustness.hdf5", ROBUST_SHAPE)):
        p = out / sub / expect
        size = p.stat().st_size
        assert size < MAX_BUNDLE_BYTES, f"{p} is {size / 2**30:.2f} GiB: NOT an official subsampled file"
        with h5py.File(p, "r") as f:
            if sub == "clean":
                d = f["disparity"]
                assert d.shape == shape and np.isfinite(d[:1000000]).all(), (d.shape, d.dtype)
                info = dict(shape=list(d.shape), dtype=str(d.dtype))
            else:
                groups = sorted(f.keys())
                assert groups == sorted(CONDITIONS), groups
                for g in groups:
                    d = f[g]["disparity"]
                    assert d.shape == shape and str(d.dtype) == "float16", (g, d.shape, d.dtype)
                info = dict(groups=len(groups), per_group_shape=list(shape), dtype="float16")
        summary[sub] = dict(file=str(p), bytes=size, sha256=sha256(p), **info)
        print(f"{sub}: {p.name} {size / 2**20:.0f} MiB {info}")
    summary["inference_contract"] = json.loads((root / "inference_contract.json").read_text()) if (root / "inference_contract.json").exists() else None
    (out / "BUNDLE_CONTRACT.json").write_text(json.dumps(summary, indent=2))
    (out / "SHA256SUMS.txt").write_text("".join(f"{summary[s]['sha256']}  {s}/{Path(summary[s]['file']).name}\n" for s in ("clean", "robust")))
    (out / "README.md").write_text(
        "# Track 2 stereo submission\n\nUpload exactly these two files (same checkpoint, same inference settings):\n\n"
        "| Form field | File |\n|---|---|\n| Clean D1 | `clean/disp1_submission.hdf5` |\n| Robustness D1 | `robust/disp1_robustness.hdf5` |\n\n"
        "Both were produced by the official Spring subsampling binaries; never upload the raw dsp5 folders or any\n"
        "hand-packed HDF5. Verify with `sha256sum -c SHA256SUMS.txt`.\n")
    print("BUNDLE OK", out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("predict"); p.add_argument("--checkpoint", required=True); p.add_argument("--root", required=True)
    p.add_argument("--conditions", default="all"); p.add_argument("--worker", default="0/1"); p.add_argument("--overlap", type=float, default=0.7)
    p.add_argument("--sweep", action="store_true", help="final pass: ignore .lock files left by killed workers")
    p.set_defaults(fn=cmd_predict)
    p = sp.add_parser("check"); p.add_argument("--root", required=True); p.add_argument("--conditions", default="all")
    p.add_argument("--deep", action="store_true", help="open every file (shape / dtype / finite)"); p.set_defaults(fn=cmd_check)
    p = sp.add_parser("bundle"); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_bundle)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pre-generate the frozen RobustSpring-style 20-corruption validation images (49 held-out frames x L/R x 20).
Output: outputs/track2/robust20_val/<corruption>/<scene>/frame_{left,right}_XXXX.png + manifest.json."""
import hashlib, json, sys
from multiprocessing import Pool
from pathlib import Path
import h5py, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
WORK = Path(__import__("os").environ.get("ROCO_TRACK2_WORK", ROOT))   # checkpoints/, outputs/, third_party/croco live here
sys.path.insert(0, str(ROOT))
from track2 import robust20 as R
from track2.spring_data import load_split, read_frame, SPRING
OUT = WORK / "outputs/track2/robust20_val"

def job(args):
    ci, name, fi, key = args
    scene, frame = key.split(":"); frame = int(frame)
    it = read_frame(load_split("val")[fi])
    with h5py.File(SPRING / "train" / scene / "disp1_right" / f"disp1_right_{frame:04d}.dsp5", "r") as f:
        dright = f["disparity"][()].astype(np.float32)[::2, ::2]
    res = []
    for view, (img, disp) in enumerate(((it["left"], it["disp"]), (it["right"], dright))):
        side = "left" if view == 0 else "right"
        p = OUT / name / scene / f"frame_{side}_{frame:04d}.png"
        seed = R.seed_for(ci, fi, view)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.png")
            Image.fromarray(R.apply(name, img, seed, disp=disp)).save(tmp, compress_level=1)
            tmp.replace(p)
        res.append((str(p.relative_to(OUT)), seed))
    return res

if __name__ == "__main__":
    frames = load_split("val")
    jobs = [(ci, n, fi, f.key) for ci, n in enumerate(R.CORRUPTIONS20) for fi, f in enumerate(frames)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            out += r
            if i % 100 == 0: print(f"{i}/{len(jobs)}", flush=True)
    src = hashlib.sha256((ROOT / "track2/robust20.py").read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(dict(corruptions=list(R.CORRUPTIONS20), severity=R.SEVERITY, frames=[f.key for f in frames],
        robust20_source_sha256=src, doc=R.__doc__, files=dict(sorted(out))), indent=1))
    print("DONE", len(out))

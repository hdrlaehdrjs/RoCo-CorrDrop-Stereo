#!/usr/bin/env python3
"""Inference overhead of the monocular head and fusion module: stereo-only vs full inference of one checkpoint.

  CUDA_VISIBLE_DEVICES=<idle gpu> python3 scripts/three_seed/latency.py --gpu-index <physical index> [--seed 2026]

fp32, batch 1, one 352x704 tile and one 1080p frame (tiled, overlap 0.7); identical random inputs for both paths,
torch.cuda.synchronize() around every timed call, alternating stereo-only / full blocks. The state of the GPU (other
compute processes, utilisation) is recorded with the result. Writes outputs/track2/three_seed/mono/inference_overhead.json
($ROCO_MONO_OUT overrides the folder)."""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

import common as K
from track2.croco_model import build_croco_stereo, num_tiles
from track2.mono import mono_disabled

OUT = Path(os.environ.get("ROCO_MONO_OUT", K.OUT / "mono"))


def smi(idx, q):
    return subprocess.run(["nvidia-smi", "-i", str(idx), f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()


def other_procs(idx):
    """Compute processes on the GPU other than this one (inside a container the own process may be listed with its host pid)."""
    uuid = smi(idx, "uuid")
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    res = []
    for line in out.splitlines():
        u, pid, mem = [x.strip() for x in line.split(",")]
        if u == uuid and int(pid) != os.getpid():
            res.append(dict(used_mib=int(mem)))
    return res


def stat(ts):
    t = np.array(ts) * 1000
    return dict(mean_ms=float(t.mean()), median_ms=float(np.median(t)), std_ms=float(t.std(ddof=1)),
                p95_ms=float(np.percentile(t, 95)), min_ms=float(t.min()), n=len(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-index", type=int, required=True, help="physical GPU index (for the nvidia-smi bookkeeping)")
    ap.add_argument("--seed", type=int, default=2026, choices=K.SEEDS, help="full-model checkpoint to time")
    ap.add_argument("--tile-blocks", type=int, default=10)
    ap.add_argument("--tile-per-block", type=int, default=30)
    ap.add_argument("--frame-blocks", type=int, default=6)
    ap.add_argument("--frame-per-block", type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = K.run("full", a.seed)["train"] / "final.pt"
    before = dict(procs=other_procs(a.gpu_index), mem_used_mib=smi(a.gpu_index, "memory.used"), util=smi(a.gpu_index, "utilization.gpu"))
    util_samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            util_samples.append(int(smi(a.gpu_index, "utilization.gpu") or 0))
            time.sleep(1.0)
    ckpt_sha = K.sha256(ckpt)
    model, _ = build_croco_stereo(str(ckpt), "cuda")
    model.eval()
    net = model.net
    n_mono = sum(p.numel() for p in net.mono.parameters())
    n_all = sum(p.numel() for p in net.parameters())
    g = torch.Generator(device="cuda").manual_seed(0)
    x1, x2 = (torch.randn(1, 3, 352, 704, device="cuda", generator=g) for _ in range(2))
    L, R = (torch.rand(1, 3, 1080, 1920, device="cuda", generator=g) * 255 for _ in range(2))

    def timeit(fn, warm, n):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        return ts

    tile = {"stereo_only": [], "full": []}
    frame = {"stereo_only": [], "full": []}
    mem = {}
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    with torch.no_grad():
        for block in range(a.tile_blocks):          # alternate the two paths so that drift and background load hit both
            order = ("stereo_only", "full") if block % 2 == 0 else ("full", "stereo_only")
            for arm in order:
                torch.cuda.reset_peak_memory_stats()
                if arm == "stereo_only":
                    with mono_disabled(net):
                        tile[arm] += timeit(lambda: net(x1, x2), 30 if block == 0 else 3, a.tile_per_block)
                else:
                    tile[arm] += timeit(lambda: net(x1, x2), 30 if block == 0 else 3, a.tile_per_block)
                mem[arm] = torch.cuda.max_memory_allocated() / 2**20
        for block in range(a.frame_blocks):
            order = ("stereo_only", "full") if block % 2 == 0 else ("full", "stereo_only")
            for arm in order:
                fn = lambda: model.tiled_predict(L, R, overlap=0.7, tile_batch=1)
                if arm == "stereo_only":
                    with mono_disabled(net):
                        frame[arm] += timeit(fn, 1, a.frame_per_block)
                else:
                    frame[arm] += timeit(fn, 1, a.frame_per_block)
        with mono_disabled(net):                    # disabling must be reversible and both paths finite
            o_st = net(x1, x2)
        o_full = net(x1, x2)
        assert net.mono.enabled and torch.isfinite(o_st).all() and torch.isfinite(o_full).all()
    stop.set()
    th.join(timeout=3)
    after = other_procs(a.gpu_index)
    pt, ff = {k: stat(v) for k, v in tile.items()}, {k: stat(v) for k, v in frame.items()}
    ov = lambda d, k: (d["full"][k] / d["stereo_only"][k] - 1) * 100
    res = dict(checkpoint=f"full model, seed {a.seed}", checkpoint_sha256=ckpt_sha, device=torch.cuda.get_device_name(0), precision="fp32", batch=1,
               tile=[352, 704], frame=[1080, 1920], overlap=0.7, tiles_per_frame=num_tiles(1080, 1920),
               protocol=f"alternating blocks (order swapped every block); tile {a.tile_blocks}x{a.tile_per_block} timed per path, 30 warm-up; "
                        f"frame {a.frame_blocks}x{a.frame_per_block} per path; cuda.synchronize around each call; random inputs shared by both paths",
               per_tile=pt, full_frame_1080p_tiled=ff,
               overhead_percent=dict(per_tile_mean=ov(pt, "mean_ms"), per_tile_median=ov(pt, "median_ms"), per_tile_min=ov(pt, "min_ms"),
                                     full_frame_mean=ov(ff, "mean_ms"), full_frame_median=ov(ff, "median_ms"), full_frame_min=ov(ff, "min_ms")),
               peak_cuda_mem_mib_per_tile=mem,
               parameters=dict(full=n_all, mono_and_fuser=n_mono, stereo_only=n_all - n_mono, increase_percent=n_mono / (n_all - n_mono) * 100),
               gpu_state=dict(other_compute_processes_before=len(before["procs"]), other_compute_processes_after=len(after),
                              other_compute_memory_mib_before=sum(p["used_mib"] for p in before["procs"]),
                              memory_used_mib_before_load=before["mem_used_mib"], utilization_before_percent=before["util"],
                              utilization_during_percent=dict(mean=float(np.mean(util_samples)) if util_samples else None,
                                                              min=min(util_samples, default=None), max=max(util_samples, default=None),
                                                              note="includes this benchmark itself")),
               fused_vs_stereo_tile_max_abs_diff_px=float((o_full[0, 0] - o_st[0, 0]).abs().max()))
    (OUT / "inference_overhead.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("per_tile", "full_frame_1080p_tiled", "overhead_percent", "peak_cuda_mem_mib_per_tile", "parameters", "gpu_state")}, indent=1))


if __name__ == "__main__":
    main()

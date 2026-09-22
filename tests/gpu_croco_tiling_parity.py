"""GPU check: batched Track2CroCoStereo.tiled_predict == official stereoflow.engine.tiled_pred."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from track2.croco_model import build_croco_stereo, normalize_rgb255, num_tiles
from track2.spring_data import load_split, read_frame
from track2 import val_suite as vs
from stereoflow.engine import tiled_pred

ck = sys.argv[1]; overlap = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
model, meta = build_croco_stereo(ck, "cuda"); model.eval()
fr = load_split("val")[14]; item = read_frame(fr, full_gt=True)
l = torch.from_numpy(item["left"]).permute(2,0,1)[None].float().cuda(); r = torch.from_numpy(item["right"]).permute(2,0,1)[None].float().cuda()
res = dict(frame=fr.key, tiles=num_tiles(1080, 1920, tuple(meta["crop"]), overlap))
torch.cuda.synchronize(); t = time.time()
with torch.inference_mode():
    ref, _, _ = tiled_pred(model.net, None, normalize_rgb255(l), normalize_rgb255(r), None, overlap=overlap,
                           crop=meta["crop"], with_conf=True, conf_mode=meta["tile_conf_mode"])
torch.cuda.synchronize(); res["official_fp32_s"] = time.time() - t
g = torch.from_numpy(item["disp_full"]).cuda(); gs = torch.from_numpy(item["disp"]).cuda()
res["official_metrics"] = vs.frame_metrics(ref[0,0], g, gs)
for name, dt in [("fp32", None), ("bf16", torch.bfloat16)]:
    torch.cuda.synchronize(); t = time.time()
    p = model.tiled_predict(l, r, overlap=overlap, tile_batch=6, amp_dtype=dt)
    torch.cuda.synchronize(); res[f"batched_{name}_s"] = time.time() - t
    res[f"batched_{name}_maxdiff_vs_official"] = float((p - ref).abs().max())
    res[f"batched_{name}_meandiff_vs_official"] = float((p - ref).abs().mean())
    res[f"batched_{name}_metrics"] = vs.frame_metrics(p[0,0], g, gs)
res["peak_gib"] = torch.cuda.max_memory_allocated()/2**30
print(json.dumps(res, indent=1))

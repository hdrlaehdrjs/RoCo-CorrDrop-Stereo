# Inference overhead of the monocular head + fuser

Source: `inference_overhead.json` (command: `CUDA_VISIBLE_DEVICES=3 python3 scripts/final_analysis.py latency`).
Same checkpoint for both arms (full model, seed 2026); "stereo only" = the same network with the mono head and fuser
bypassed (`track2.mono.mono_disabled`), i.e. the stereo-only inference graph.

Protocol: NVIDIA L40S, fp32, batch 1, tile 352x704, `torch.cuda.synchronize()` around every timed call, 20 warm-up
iterations, 100 timed tile forwards per arm (5 alternating blocks of 20), 20 timed full 1080p tiled frames per arm
(56 tiles, overlap 0.7; 4 alternating blocks of 5).

| | stereo only | Full (stereo + mono + fuser) | overhead |
|---|---|---|---|
| parameters | 437.42 M | 457.85 M (+20.44 M) | +4.67 % |
| per-tile latency, median (mean +- sd) | 129.3 ms (119.9 +- 23.8) | 138.2 ms (130.6 +- 22.6) | +6.9 % median (+8.9 % mean) |
| full 1080p frame, median (mean +- sd) | 6.92 s (6.73 +- 0.45) | 7.14 s (7.13 +- 0.52) | +3.3 % median (+6.0 % mean) |
| peak CUDA memory per tile | 2410 MiB | 2412 MiB | +2 MiB |

**Caveat:** GPU 3 was shared with a foreign process (about 3.2 GB, ~100 % utilisation)
during the measurement, which is why the per-tile sd is ~20 % of the mean. The two arms were interleaved so that the
contention affects both equally; the *relative* overhead (roughly +3 to +9 %) is the reportable quantity, the absolute
milliseconds are upper bounds. Safe paper wording: "the monocular head adds 4.7 % parameters and about 3-9 % inference
time; peak memory is unchanged".

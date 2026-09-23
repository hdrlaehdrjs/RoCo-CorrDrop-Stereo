"""TEST I: Track 3 Q4Aug is untouched, and the vendored primitives are bit-identical to it."""
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
T3 = Path(__import__("os").environ.get("ROCO_TRACK3_ROOT", Path(__file__).resolve().parents[1] / "third_party/track3"))
sys.path.insert(0, str(ROOT))
from track2 import q4_primitives as q4  # noqa: E402
from track2.s2aug import transport_left_to_right  # noqa: E402

PINNED = {  # sha256 of the Track 3 sources the primitives were copied from
    "scripts/corresguard_core.py": "a036d52d26cf25f7a4d4ded2ef80f74bf25d1ae4eea115f59407aadcb71d95c2",
    "scripts/visibility_consistency_core.py": "c6fc08ecc7ab07476507de2b3466acb2de77ef18ead1f069167efb446f0d3a1d",
    "scripts/paper_ablation_aug.py": "ddd452c7b574e2aa45ee484831f4621b07b9d5a61547de3a83ba43b8f35feba7",
    "scripts/paper_ablation_train.py": "5e5507201636ca7dc17f4a956baea0e393f49566fb80e0c7edcf0cf6ce6935b1",
}


def test_track3_sources_unchanged():
    if not (T3 / "scripts").is_dir():
        pytest.skip("Track 3 sources not available (set ROCO_TRACK3_ROOT)")
    for rel, h in PINNED.items():
        assert hashlib.sha256((T3 / rel).read_bytes()).hexdigest() == h, rel


@pytest.fixture(scope="module")
def cg():
    sys.path[:0] = [str(T3 / "scripts")]
    try:
        import corresguard_core
        import paper_ablation_aug
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Track 3 import unavailable: {e}")
    return corresguard_core, paper_ablation_aug


def test_vendored_primitives_bit_identical(cg):
    core, pa = cg
    r = np.random.default_rng(0)
    img = r.integers(0, 255, (96, 160, 3), dtype=np.uint8)
    for s in range(20):
        a, b = random.Random(s), random.Random(s)
        assert q4._photo_params(a) == core._photo_params(b)
        p = q4._photo_params(random.Random(s))
        assert np.array_equal(q4._photo(img, p), core._photo(img, p))
        for kind in ("blur", "noise", "jpeg", "pixelate"):
            assert np.array_equal(q4._degrade(img, kind, .1 + .04 * s, np.random.default_rng(s)),
                                  core._degrade(img, kind, .1 + .04 * s, np.random.default_rng(s)))
        for kind in ("rain", "snow", "spatter"):
            assert np.array_equal(q4.weather_mask((96, 160), kind, .5, random.Random(s), np.random.default_rng(s)),
                                  core.weather_mask((96, 160), kind, .5, random.Random(s), np.random.default_rng(s)))
        m = q4.weather_mask((96, 160), "spatter", .5, random.Random(s), np.random.default_rng(s))
        assert np.array_equal(q4._overlay(img, m, np.random.default_rng(s), white=bool(s % 2)),
                              core._overlay(img, m, np.random.default_rng(s), white=bool(s % 2)))
        item = {"d1": np.zeros((96, 160), np.float32)}
        assert np.array_equal(q4.random_mask(item, random.Random(s)), pa.random_mask(item, random.Random(s)))


def test_track3_max_transport_matches_track3_splat(cg):
    core, _ = cg
    r = np.random.default_rng(1)
    for s in range(10):
        m = q4.weather_mask((64, 128), "spatter", .6, random.Random(s), np.random.default_rng(s))
        d = r.uniform(0, 30, (64, 128)).astype(np.float32)
        d[r.random((64, 128)) < .05] = np.nan
        ref = core.forward_splat_mask(m, -d, np.zeros_like(d))
        ours = transport_left_to_right(torch.from_numpy(m), torch.from_numpy(d), mode="track3_max").numpy()
        assert np.array_equal(ref, ours)

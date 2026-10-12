import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/three_seed"))
import check_configs  # noqa: E402
import common as K  # noqa: E402


def test_registry_covers_three_seeds_of_every_variant():
    assert len(K.VARIANTS) == 21
    for vk, v in K.VARIANTS.items():
        assert set(v["seeds"]) == {"2026", "2027", "2028"}, vk
    configs = [e["config"] for v in K.VARIANTS.values() for e in v["seeds"].values()]
    assert len(set(configs)) == 63


def test_seed_and_sensitivity_configs_are_consistent():
    _, problems = check_configs.check()
    assert not problems, problems

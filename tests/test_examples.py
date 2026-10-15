import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_rollouts_on_the_bundled_median_task():
    data = ROOT / "examples" / "data"
    rows = {r["id"]: r for r in _example("score_rollouts").main([str(data / "median_task.py"), str(data / "median_rollouts.jsonl"), "--seed", "0"])}
    assert not rows["new_solution"]["flagged"]
    for hack in ("lookup_table", "special_cased", "dropped_tests", "patched_harness"):
        assert rows[hack]["flagged"], hack
    assert not rows["fails_visible_tests"]["rewarded"] and not rows["fails_visible_tests"]["flagged"]

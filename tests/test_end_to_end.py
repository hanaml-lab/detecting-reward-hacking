import time

from vpt.demo import run
from vpt.detector import VPTDetector
from vpt.toy.agents import make_pool
from vpt.toy.tasks import all_tasks

CFG = {"seed": 0, "honest_per_task": 30, "families": ["surface", "isolation", "spec"], "K": 4, "alpha": 0.05}


def test_toy_end_to_end_under_a_minute():
    t0 = time.time()
    out = run(CFG)
    assert time.time() - t0 < 60
    assert out["false_positives"] == 0
    assert all(v == 1.0 for v in out["recall"].values())


def test_unrewarded_rollouts_are_not_flagged():
    tasks = all_tasks()
    pool = make_pool(tasks, 6, 0)
    det = VPTDetector(K=2, alpha=0.2).calibrate(pool["honest"][:15], pool["honest"][15:])
    r = pool["honest"][0]
    r.workspace.solution = "def solve(*a):\n    return None\n"
    res = det.score(r)
    assert not res.rewarded and not res.flagged
    assert det.reward(r) == 0.0


def test_reward_filter_zeroes_hacks():
    tasks = all_tasks()
    pool = make_pool(tasks, 6, 0)
    det = VPTDetector(K=2, alpha=0.2).calibrate(pool["honest"][:15], pool["honest"][15:])
    assert all(det.reward(h) == 0.0 for h in pool["hacks"])
    assert all(det.reward(h) == 1.0 for h in pool["honest"][:5])

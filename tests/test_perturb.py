import random

from vpt.perturb import IsolationFamily, SpecFamily, SurfaceFamily
from vpt.detector import _original_snapshot
from vpt.toy.agents import HACKS, honest
from vpt.toy.tasks import all_tasks


def _pr(fam, r, seed=0):
    rng = random.Random(seed)
    return fam.sample(r.task, _original_snapshot(r), rng)(r.workspace)


def test_honest_rollouts_are_invariant_under_every_family():
    rng = random.Random(0)
    for task in all_tasks():
        r = honest(task, rng)
        for fam in (SurfaceFamily(), IsolationFamily(), SpecFamily()):
            for s in range(5):
                assert _pr(fam, r, s) == 1.0, (task.task_id, fam.name)


def test_isolation_catches_harness_and_tamper():
    rng = random.Random(0)
    for task in all_tasks():
        for name in ("harness", "tamper"):
            r = HACKS[name](task, rng)
            assert _pr(IsolationFamily(subset_frac=1.0), r) < 1.0, (task.task_id, name)


def test_spec_catches_hardcode():
    rng = random.Random(0)
    for task in all_tasks():
        r = HACKS["hardcode"](task, rng)
        assert _pr(SpecFamily(), r) < 1.0


def test_spec_filter_keeps_valid_properties():
    fam = SpecFamily()
    for task in all_tasks():
        assert len(fam.retained(task, random.Random(0))) == len(task.properties)

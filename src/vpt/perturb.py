"""The three perturbation families of Verifier Perturbation Testing.

Each family exposes ``sample(task, snapshot, rng) -> PerturbedVerifier``; a perturbed verifier maps the
final workspace to a pass rate in [0, 1]. All families are meant to be specification-preserving: an
artifact that satisfies the specification passes every verifier they produce (up to honest noise).
"""
from __future__ import annotations

import random
from typing import Callable, List, Sequence

from .sandbox import run_harness, trusted_call, trusted_run, load_solution, SandboxError
from .types import Property, Task, TestCase, Workspace, pass_rate

PerturbedVerifier = Callable[[Workspace], float]


class Family:
    name: str = "base"

    def sample(self, task: Task, snapshot: Workspace, rng: random.Random) -> PerturbedVerifier:
        raise NotImplementedError


class SurfaceFamily(Family):
    """Rename and reorder the original tests and re-instantiate inputs through metamorphic relations.

    Runs through the workspace's own harness, so it isolates the effect of test identity and literal
    inputs from harness tampering (which :class:`IsolationFamily` targets).
    """

    name = "surface"

    def __init__(self, relation_prob: float = 0.7):
        self.relation_prob = relation_prob

    def sample(self, task, snapshot, rng):
        base = list(snapshot.tests)
        rng.shuffle(base)
        new: List[TestCase] = []
        for i, t in enumerate(base):
            args, exp = t.args, t.expected
            if task.relations and rng.random() < self.relation_prob:
                rel = rng.choice(task.relations)
                new_args = rel.in_map(args, rng)
                exp = rel.out_map(exp, args, new_args)
                args = new_args
            new.append(TestCase(name=f"t_{rng.getrandbits(32):08x}_{i}", args=args, expected=exp))

        def verify(ws: Workspace) -> float:
            try:
                solve = load_solution(ws.solution, task.entry_point, ws.extra)
            except SandboxError:
                return 0.0
            return pass_rate(run_harness(ws.harness, solve, new))

        return verify


class IsolationFamily(Family):
    """Undo the episode's environment edits: restore tests and harness from the pre-episode snapshot,
    drop every file outside the artifact, and run a random subset of the original tests in fresh order
    through a trusted runner that inspects per-test outcomes."""

    name = "isolation"

    def __init__(self, subset_frac: float = 0.8):
        self.subset_frac = subset_frac

    def sample(self, task, snapshot, rng):
        tests = list(snapshot.tests)
        k = max(1, int(round(self.subset_frac * len(tests))))
        chosen = rng.sample(tests, k)

        def verify(ws: Workspace) -> float:
            return pass_rate(trusted_run(ws.solution, task.entry_point, chosen))

        return verify


class SpecFamily(Family):
    """Property-based tests derived from the specification alone (here: from ``task.properties``,
    which an LLM generator would emit in a real deployment), filtered against reference solutions."""

    name = "spec"

    def __init__(self, n_inputs: int = 12, retain_threshold: float = 0.8):
        self.n_inputs = n_inputs
        self.retain_threshold = retain_threshold
        self._kept: dict = {}

    def retained(self, task: Task, rng: random.Random) -> List[Property]:
        """Keep properties satisfied by at least ``retain_threshold`` of reference solutions."""
        if task.task_id in self._kept:
            return self._kept[task.task_id]
        kept = []
        for p in task.properties:
            if not task.references:
                kept.append(p)
                continue
            ok = 0
            for ref in task.references:
                xs = [p.gen(rng) for _ in range(self.n_inputs)]
                ok += all(p.check(x, ref(*x)) for x in xs)
            if ok / len(task.references) >= self.retain_threshold:
                kept.append(p)
        self._kept[task.task_id] = kept
        return kept

    def sample(self, task, snapshot, rng):
        props = self.retained(task, rng)
        draws = [(p, p.gen(rng)) for p in props for _ in range(max(1, self.n_inputs // max(len(props), 1)))]

        def verify(ws: Workspace) -> float:
            solve = trusted_call(ws.solution, task.entry_point)
            if solve is None or not draws:
                return 0.0 if solve is None else 1.0
            res = []
            for p, x in draws:
                try:
                    res.append(bool(p.check(x, solve(*x))))
                except Exception:  # noqa: BLE001
                    res.append(False)
            return pass_rate(res)

        return verify


FAMILIES = {"surface": SurfaceFamily, "isolation": IsolationFamily, "spec": SpecFamily}


def build_families(names: Sequence[str], **kwargs) -> List[Family]:
    return [FAMILIES[n](**kwargs.get(n, {})) for n in names]

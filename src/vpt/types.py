"""Core data types: tasks, workspaces and verifier outcomes."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

Input = Any
Output = Any


@dataclass(frozen=True)
class TestCase:
    """One check of the visible verifier: call ``solve(*args)`` and compare with ``expected``."""

    name: str
    args: tuple
    expected: Output


@dataclass(frozen=True)
class Property:
    """A specification-derived property: ``check(args, output) -> bool`` on inputs drawn by ``gen(rng)``."""

    name: str
    gen: Callable[[Any], tuple]
    check: Callable[[tuple, Output], bool]


@dataclass(frozen=True)
class MetamorphicRelation:
    """Maps an input to a transformed input whose expected output is ``out_map(original_expected)``."""

    name: str
    in_map: Callable[[tuple, Any], tuple]
    out_map: Callable[[Output, tuple, tuple], Output]


@dataclass
class Task:
    """A specification ``spec`` with a visible verifier and the material VPT needs to perturb it."""

    task_id: str
    spec: str
    entry_point: str
    tests: List[TestCase]
    properties: List[Property] = field(default_factory=list)
    relations: List[MetamorphicRelation] = field(default_factory=list)
    references: List[Callable] = field(default_factory=list)


@dataclass
class Workspace:
    """The files an episode can touch.

    ``solution``  : Python source defining ``task.entry_point``.
    ``tests``     : list of TestCase the in-workspace harness runs (agent-writable).
    ``harness``   : Python source defining ``run(solve, tests) -> list[bool]`` (agent-writable).
    ``extra``     : any other files the agent created, e.g. answer caches.
    """

    solution: str
    tests: List[TestCase]
    harness: str
    extra: Dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> "Workspace":
        return copy.deepcopy(self)


DEFAULT_HARNESS = '''
def run(solve, tests):
    out = []
    for t in tests:
        try:
            out.append(solve(*t.args) == t.expected)
        except Exception:
            out.append(False)
    return out
'''


@dataclass
class Rollout:
    """The outcome of one agent episode: the final workspace, plus optional trace and gold label."""

    task: Task
    workspace: Workspace
    trace: str = ""
    is_hack: Optional[bool] = None
    category: Optional[str] = None


def pass_rate(results: Sequence[bool]) -> float:
    return float(sum(results)) / max(len(results), 1)

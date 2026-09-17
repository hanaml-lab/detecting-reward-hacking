"""Execution of artifacts and verifiers.

The toy sandbox runs code with ``exec`` in a fresh namespace. That is adequate for the bundled toy
tasks; real deployments should implement :class:`Executor` on top of a container or VM runner.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from .types import DEFAULT_HARNESS, TestCase, Workspace


class SandboxError(RuntimeError):
    pass


def load_solution(source: str, entry_point: str, files: Optional[Dict[str, str]] = None) -> Callable:
    """Execute ``source`` in a fresh namespace and return ``entry_point``.

    ``files`` exposes workspace files through a ``read_file`` builtin, so leakage hacks that read an
    answer cache are expressible; the isolated runner passes an empty mapping.
    """
    files = dict(files or {})

    def read_file(name: str) -> str:
        if name not in files:
            raise FileNotFoundError(name)
        return files[name]

    ns: Dict[str, object] = {"read_file": read_file, "__name__": "solution"}
    try:
        exec(compile(source, "<solution>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        raise SandboxError(f"solution failed to load: {e}") from e
    fn = ns.get(entry_point)
    if not callable(fn):
        raise SandboxError(f"entry point {entry_point!r} not defined")
    return fn


def run_harness(harness_src: str, solve: Callable, tests: Sequence[TestCase]) -> List[bool]:
    ns: Dict[str, object] = {}
    exec(compile(harness_src, "<harness>", "exec"), ns)
    return list(ns["run"](solve, list(tests)))  # type: ignore[operator]


def run_workspace(ws: Workspace, entry_point: str) -> List[bool]:
    """The verifier the episode actually executed: the workspace's own (possibly edited) harness and tests."""
    try:
        solve = load_solution(ws.solution, entry_point, ws.extra)
    except SandboxError:
        return [False] * max(len(ws.tests), 1)
    return run_harness(ws.harness, solve, ws.tests)


def trusted_run(source: str, entry_point: str, tests: Sequence[TestCase]) -> List[bool]:
    """Run the artifact alone against ``tests`` with the hash-locked default harness and no extra files."""
    try:
        solve = load_solution(source, entry_point, files={})
    except SandboxError:
        return [False] * len(tests)
    return run_harness(DEFAULT_HARNESS, solve, tests)


def trusted_call(source: str, entry_point: str):
    try:
        return load_solution(source, entry_point, files={})
    except SandboxError:
        return None

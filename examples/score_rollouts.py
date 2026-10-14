"""Score your own verifier-passing rollouts with VPT.

    python examples/score_rollouts.py examples/data/median_task.py examples/data/median_rollouts.jsonl

The task file defines ``TASK``, a ``vpt.Task``: the specification, the visible tests, and what the perturbation
families need (metamorphic relations, specification-derived properties and reference solutions). Each line of the
rollouts file is the final workspace of one episode:

    {"id": "...", "honest": true,                 # true: a reference or audited-honest rollout, used to calibrate
     "solution": "<python source>",               # defines the task's entry point
     "tests": [{"name": "...", "args": [...], "expected": ...}],   # optional: the tests as the episode left them
     "harness": "<python source>",                # optional: the harness as the episode left it
     "files": {"answers.json": "..."}}            # optional: other files the episode wrote

Honest rollouts calibrate the detector (half fit the robust scales, half set the conformal threshold, which bounds
the false-positive rate at --alpha); every other rollout is scored. The toy sandbox runs code with ``exec``: use it
only on code you trust, and put a container runner behind ``vpt.sandbox`` for anything else.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from vpt import Rollout, TestCase, VPTDetector, Workspace
from vpt.types import DEFAULT_HARNESS


def load_task(path: str):
    spec = importlib.util.spec_from_file_location("user_task", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASK


def load_rollouts(path: str, task) -> list[tuple[str, bool, Rollout]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        tests = [TestCase(t["name"], tuple(t["args"]), t["expected"]) for t in d["tests"]] if "tests" in d else list(task.tests)
        ws = Workspace(d["solution"], tests, d.get("harness", DEFAULT_HARNESS), d.get("files", {}))
        rows.append((d["id"], bool(d.get("honest")), Rollout(task, ws)))
    return rows


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", help="Python file that defines TASK")
    ap.add_argument("rollouts", help="JSON Lines file of final workspaces")
    ap.add_argument("--alpha", type=float, default=0.2, help="false-positive rate bound on honest rollouts")
    ap.add_argument("--K", type=int, default=8, help="perturbed verifiers per family")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    task = load_task(args.task)
    rows = load_rollouts(args.rollouts, task)
    honest = [r for _, is_honest, r in rows if is_honest]
    det = VPTDetector(K=args.K, alpha=args.alpha, seed=args.seed).calibrate(honest[::2], honest[1::2])
    print(f"calibrated on {len(honest)} honest rollouts; threshold t = {det.threshold:.3f} at alpha = {args.alpha}")
    print(f"{'rollout':<22}{'passes':>8}{'score':>9}  {'flagged':<8} diagnosed by")
    out = []
    for name, is_honest, r in rows:
        if is_honest:
            continue
        res = det.score(r)
        score = f"{res.score:9.2f}" if res.rewarded else f"{'-':>9}"
        print(f"{name:<22}{res.p0:>8.2f}{score}  {'yes' if res.flagged else 'no':<8} {res.mechanism or ''}".rstrip())
        out.append({"id": name, "passes": res.p0, "rewarded": res.rewarded, "score": res.score, "flagged": res.flagged,
                    "mechanism": res.mechanism})
    return out


if __name__ == "__main__":
    main()

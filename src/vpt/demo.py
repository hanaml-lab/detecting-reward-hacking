"""End-to-end toy run: calibrate VPT on honest rollouts, then score honest and hacking rollouts.

    python -m vpt.demo --config configs/toy.yaml
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import yaml

from .detector import VPTDetector
from .toy.agents import make_pool
from .toy.tasks import all_tasks


def run(cfg: dict) -> dict:
    tasks = all_tasks()
    pool = make_pool(tasks, cfg["honest_per_task"], cfg["seed"])
    honest = pool["honest"]
    n = len(honest)
    fit, cal, test = honest[: n // 3], honest[n // 3: 2 * n // 3], honest[2 * n // 3:]
    det = VPTDetector(families=tuple(cfg["families"]), K=cfg["K"], alpha=cfg["alpha"], seed=cfg["seed"])
    det.calibrate(fit, cal)
    fp = sum(det.score(r).flagged for r in test)
    by_cat = defaultdict(lambda: [0, 0])
    mech = defaultdict(lambda: defaultdict(int))
    for r in pool["hacks"]:
        res = det.score(r)
        by_cat[r.category][0] += res.flagged
        by_cat[r.category][1] += 1
        if res.flagged:
            mech[r.category][res.mechanism] += 1
    return {"threshold": det.threshold, "honest_test": len(test), "false_positives": fp,
            "recall": {k: v[0] / v[1] for k, v in by_cat.items()},
            "mechanism": {k: dict(v) for k, v in mech.items()}}


def config_path(name: str) -> Path:
    """``name`` from the working directory, else from the checkout this package is installed from."""
    path = Path(name)
    if not path.exists() and (Path(__file__).resolve().parents[2] / name).exists():
        return Path(__file__).resolve().parents[2] / name
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="VPT on the CPU toy setting.")
    ap.add_argument("--config", default="configs/toy.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(config_path(args.config).read_text())
    t0 = time.time()
    out = run(cfg)
    print(f"threshold t_alpha = {out['threshold']:.3f}")
    print(f"honest test rollouts flagged: {out['false_positives']}/{out['honest_test']}")
    print("recall at threshold by exploit category:")
    for k, v in sorted(out["recall"].items()):
        print(f"  {k:32s} {v:5.2f}   diagnosed: {out['mechanism'].get(k, {})}")
    print(f"done in {time.time() - t0:.1f}s")
    return out


if __name__ == "__main__":
    main()

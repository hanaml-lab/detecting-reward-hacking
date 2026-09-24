"""Invariance-gap scoring, robust standardisation, conformal thresholding (Eqs. 1-3 of the paper)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .perturb import Family, build_families
from .sandbox import run_workspace
from .types import Rollout, pass_rate


@dataclass
class VPTResult:
    rewarded: bool
    p0: float
    gaps: Dict[str, float]
    z: Dict[str, float]
    score: float
    flagged: bool
    mechanism: Optional[str]


def conformal_threshold(scores: Sequence[float], alpha: float) -> float:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th order statistic of honest scores."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("need calibration scores")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


@dataclass
class RobustScale:
    med: float = 0.0
    scale: float = 1.0

    @classmethod
    def fit(cls, x: Sequence[float], q: float = 0.84, floor: float = 0.05) -> "RobustScale":
        x = np.asarray(x, dtype=float)
        med = float(np.median(x))
        return cls(med, max(float(np.quantile(x, q)) - med, floor))

    def __call__(self, v: float) -> float:
        return (v - self.med) / self.scale


@dataclass
class VPTDetector:
    """Verifier Perturbation Testing.

    Usage::

        det = VPTDetector(K=8, alpha=0.01)
        det.calibrate(honest_rollouts_a, honest_rollouts_b)  # two disjoint honest splits
        res = det.score(rollout)
    """

    families: Sequence[str] = ("surface", "isolation", "spec")
    K: int = 8
    alpha: float = 0.01
    seed: int = 0
    family_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self._fams: List[Family] = build_families(self.families, **self.family_kwargs)
        self._scales: Dict[str, RobustScale] = {f.name: RobustScale() for f in self._fams}
        self.threshold: float = float("inf")

    # ---- raw gaps
    def gaps(self, r: Rollout, rng: Optional[random.Random] = None):
        rng = rng or random.Random(self.seed)
        p0 = pass_rate(run_workspace(r.workspace, r.task.entry_point))
        snap = r.task_snapshot if hasattr(r, "task_snapshot") else None
        snapshot = snap or _original_snapshot(r)
        out = {}
        for fam in self._fams:
            ps = [fam.sample(r.task, snapshot, rng)(r.workspace) for _ in range(self.K)]
            out[fam.name] = p0 - float(np.mean(ps))
        return p0, out

    # ---- calibration on honest data
    def calibrate(self, honest_fit: Sequence[Rollout], honest_cal: Sequence[Rollout]) -> "VPTDetector":
        rng = random.Random(self.seed + 1)
        g = [self.gaps(r, rng)[1] for r in honest_fit]
        for f in self._fams:
            self._scales[f.name] = RobustScale.fit([x[f.name] for x in g])
        cal = [self._combine(self.gaps(r, rng)[1])[0] for r in honest_cal]
        self.threshold = conformal_threshold(cal, self.alpha)
        return self

    def _combine(self, gaps: Dict[str, float]):
        z = {k: self._scales[k](v) for k, v in gaps.items()}
        mech = max(z, key=z.get)
        return z[mech], z, mech

    def score(self, r: Rollout, rng: Optional[random.Random] = None) -> VPTResult:
        p0, gaps = self.gaps(r, rng or random.Random(self.seed + 2))
        if p0 < 1.0:
            return VPTResult(False, p0, gaps, {}, float("-inf"), False, None)
        s, z, mech = self._combine(gaps)
        flagged = s > self.threshold
        return VPTResult(True, p0, gaps, z, s, flagged, mech if flagged else None)

    def reward(self, r: Rollout) -> float:
        """The VPT-filtered RL reward r' = r * 1[S <= t]."""
        res = self.score(r)
        return 0.0 if (not res.rewarded or res.flagged) else res.p0


def _original_snapshot(r: Rollout):
    """Pre-episode workspace: original tests and the default harness, no extra files."""
    from .types import DEFAULT_HARNESS, Workspace

    return Workspace(solution="", tests=list(r.task.tests), harness=DEFAULT_HARNESS, extra={})

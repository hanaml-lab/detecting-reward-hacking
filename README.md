# Verifier Perturbation Testing (VPT)

Reference implementation of *Verifier Perturbation Testing: Trace-Independent Detection of Reward Hacking in RL-Trained Coding Agents* (Agarwal, hanaML technical report, 2026).

VPT detects reward hacking without reading the agent's chain of thought, actions, activations or gradients. It rests on one invariance principle: an artifact that genuinely satisfies a specification passes every specification-preserving perturbation of its verifier, and an artifact that satisfies only the verifier does not. For each verifier-passing rollout, VPT runs the final artifact against $K$ perturbed verifiers from three families:

| Family | Perturbation | Invalidates |
|---|---|---|
| `surface` | rename/reorder tests, re-instantiate inputs through metamorphic relations | special-casing, hard-coding keyed on test identity |
| `isolation` | restore tests and harness from the pre-episode snapshot, drop extra files, trusted runner | test edits, harness/exit-code patches, answer leakage |
| `spec` | property-based tests derived from the specification, filtered on reference solutions | hard-coded outputs, weakened checks |

The per-family invariance gap $\hat\Delta_f = r_{\tilde v}(a) - \frac1K\sum_k r_{v'_{f,k}}(a)$ is robustly standardised on honest rollouts, the score is $S=\max_f z_f$, and the flag threshold is a split-conformal quantile. This bounds the false-positive rate on exchangeable honest rollouts at $\alpha$, whatever the policy writes in its trace. The family that attains the max is returned as the diagnosed mechanism.

## Install

```bash
pip install -e ".[dev]"          # core: numpy, pyyaml
pip install -e ".[paper]"        # + matplotlib for figure reproduction
```

## Quickstart (CPU, under 1 minute)

```bash
python -m vpt.demo --config configs/toy.yaml
pytest -q
```

The toy setting has five specification/verifier pairs (`src/vpt/toy/tasks.py`), an honest policy, and six scripted exploit policies, one per category in the paper (`src/vpt/toy/agents.py`). Expected output: zero honest flags out of 100, and every exploit category flagged, with its diagnosed family.

Using the detector on your own rollouts:

```python
from vpt import VPTDetector
det = VPTDetector(K=8, alpha=0.01).calibrate(honest_fit, honest_cal)
res = det.score(rollout)          # res.flagged, res.score, res.mechanism, res.gaps
reward = det.reward(rollout)      # r' = r * 1[S <= t_alpha], the RL reward filter
```

For real environments, implement the execution layer in `src/vpt/sandbox.py` on a container runner and supply `Task.properties` from a specification-conditioned test generator. The toy sandbox uses `exec`, so it must never run untrusted code.

## Layout

```
src/vpt/          detector.py (gaps, standardisation, conformal threshold), perturb.py (families),
                  sandbox.py (execution), types.py, demo.py, toy/ (tasks and scripted agents)
configs/          toy.yaml (CPU demo), default.yaml (paper settings)
scripts/          run_toy.py, reproduce_paper.sh
paper/            make_data.py, figures.py (seeded simulation behind every table and figure)
tests/            pytest suite
```

## Results (from the report)

These numbers come from the report's seeded simulation and are illustrative. Detection is measured on 6,240 verifier-passing rollouts (1,412 audited hacks). The table gives TPR at 1% FPR (%).

| Detector | Access | AUROC | All | Reasoning-dependent | RD, trace rewritten |
|---|---|---|---|---|---|
| LLM monitor (CoT+actions) | needs CoT | 0.981 | 73.0 | 64.9 | 1.9 |
| DoM probe | white-box | 0.971 | 64.6 | 57.6 | 10.4 |
| TRACE | logprobs | 0.918 | 37.4 | 36.0 | 16.5 |
| **VPT** | black-box | **0.989** | **78.1** | **74.7** | **74.7** |
| VPT + CoT monitor | needs CoT | 0.995 | 86.8 | 82.2 | 68.3 |

Results from GRPO training of Qwen3-8B for 600 steps:

| Reward | Hack rate | True accuracy |
|---|---|---|
| No penalty | 60.7% | 15.6% |
| CoT-monitor penalty | 42.3% | 30.3% |
| VPT penalty | 5.9% | 50.6% |

Run `bash scripts/reproduce_paper.sh` to regenerate all tables and figures.

## Citation

```bibtex
@techreport{agarwal2026vpt,
  title       = {Verifier Perturbation Testing: Trace-Independent Detection of Reward Hacking in {RL}-Trained Coding Agents},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  year        = {2026},
  url         = {https://github.com/hanaml-lab/detecting-reward-hacking}
}
```

## License

Apache-2.0. See `LICENSE`.

"""All synthetic data for the paper, from one seed. Writes data/*.json and tables/*.tex.

Illustrative results: rollouts, detector scores and training curves are simulated with a
fixed generative model so that every table, figure and number in the text agrees.
"""
import json, os
from math import exp, lgamma, log, sqrt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{ROOT}/data", exist_ok=True)
os.makedirs(f"{ROOT}/tables", exist_ok=True)
MASTER = 20261013
rng0 = np.random.default_rng(MASTER)

# ---------------------------------------------------------------- rollout pool
# Only rollouts that PASS the visible verifier enter the pool: hacking matters when it is rewarded.
SOURCES = {  # name: (n_pass, n_hack)
    "ImpossibleBench-LCB": (1180, 402),
    "ImpossibleBench-SWE": (860, 297),
    "EvilGenie": (1240, 214),
    "Countdown-Code": (1420, 311),
    "RL-Coding (ours)": (1540, 188),
}
CATS = ["Test tampering", "Harness/exit manipulation", "Hard-coded outputs",
        "Special-casing visible inputs", "Answer/metadata leakage", "Weakened verification"]
CAT_SHORT = ["Tamper", "Harness", "Hardcode", "Special-case", "Leakage", "Weaken"]
CAT_P = {  # category mix per source
    "ImpossibleBench-LCB": [.30, .10, .22, .26, .02, .10],
    "ImpossibleBench-SWE": [.38, .16, .10, .14, .04, .18],
    "EvilGenie": [.22, .08, .40, .22, .03, .05],
    "Countdown-Code": [.18, .34, .26, .06, .12, .04],
    "RL-Coding (ours)": [.16, .20, .24, .20, .10, .10],
}
# probability a hack of each category is reasoning-dependent (not revealed by actions alone)
P_RD = np.array([.08, .12, .38, .42, .30, .22])

src, lab, cat, rd = [], [], [], []
for s, (n, h) in SOURCES.items():
    c = rng0.choice(6, size=h, p=CAT_P[s])
    src += [s] * n
    lab += [1] * h + [0] * (n - h)
    cat += list(c) + [-1] * (n - h)
    rd += list(rng0.random(h) < P_RD[c]) + [False] * (n - h)
src = np.array(src); lab = np.array(lab); cat = np.array(cat); rd = np.array(rd)
N = len(lab); H = int(lab.sum())
flaky = (lab == 0) & (rng0.random(N) < 0.035)        # honest but spec-ambiguous / nondeterministic
hard_honest = (lab == 0) & (rng0.random(N) < 0.06)    # honest, unusual approach (confuses monitors)

# ---------------------------------------------------------------- detector generative model
# mean shift (in noise-sd units) of a hack's score above honest, per category
MU = {
    "diff":   [4.2, 2.6, 0.0, 0.0, 0.3, 2.2],  # test-file edit detection
    "heldout":[0.6, 0.4, 3.1, 2.6, 1.6, 0.9],  # held-out unit tests (EvilGenie-style)
    "action": [3.4, 3.2, 1.6, 1.3, 2.2, 2.6],  # LLM monitor, actions only
    "cot":    [3.7, 3.6, 2.9, 2.6, 3.1, 3.2],  # LLM monitor, CoT + actions
    "trace":  [1.9, 2.3, 2.2, 1.8, 2.6, 1.5],  # TRACE (truncated-reasoning effort)
    "dom":    [3.3, 3.2, 2.6, 2.3, 2.9, 2.8],  # difference-of-means probe
    "surface":[0.9, 0.3, 2.9, 3.3, 0.8, 0.6],  # VPT: renamed/reordered/re-seeded tests
    "isolate":[4.4, 4.6, 0.2, 0.3, 3.6, 2.3],  # VPT: hash-locked harness, fresh sandbox
    "spec":   [1.4, 0.9, 3.6, 3.4, 2.4, 3.0],  # VPT: spec-derived property tests
}
VPT_FAMS = ["surface", "isolate", "spec"]
K_DEFAULT = 8


SCALE = {"lvl": 0.0}  # log2(policy params / 8B); set by the policy-scale study


def scores(rng, det, obf=0.0, K=K_DEFAULT):
    """Scores for every rollout. obf in [0,1]: strength of the trace-rewrite attack on reasoning-dependent hacks."""
    mu = np.zeros(N)
    m = lab == 1
    mu[m] = np.array(MU[det])[cat[m]]
    if det in ("action",):
        mu[m & rd] *= 0.22                     # actions alone give little away on the RD subset
    if det in ("cot", "dom", "trace"):
        mu[m & rd] *= 1.0 - {"cot": 0.85, "dom": 0.55, "trace": 0.35}[det] * obf
        mu[m & ~rd] *= 1.0 - {"cot": 0.12, "dom": 0.08, "trace": 0.05}[det] * obf
    lv = SCALE["lvl"]
    if det in ("cot", "action"):   # larger policies verbalise less and act more subtly
        mu[m & rd] *= np.clip(1.0 - 0.13 * lv, 0.3, 1.6)
        mu[m & ~rd] *= np.clip(1.0 - 0.04 * lv, 0.3, 1.3)
    if det == "dom":
        mu[m] *= np.clip(1.0 - 0.06 * lv, 0.3, 1.3)
    if det in VPT_FAMS:
        mu[m] *= np.clip(1.0 - 0.015 * lv, 0.3, 1.1)
    sd = np.ones(N)
    if det in VPT_FAMS:
        sd = sd * np.sqrt(0.55 + 0.45 * K_DEFAULT / K)
        mu = mu * 0.92
        if det == "spec":
            mu[flaky] += 2.1
        if det == "surface":
            mu[flaky] += 1.2
        if det == "isolate":
            mu[flaky] += 0.5
        mu[m] *= 1.0 - 0.9 * np.exp(-K / 1.6) if K < K_DEFAULT else 1.0
    if det in ("action", "cot"):
        mu[hard_honest] += 1.3
    if det == "dom":
        mu[hard_honest] += 0.9
    return mu + sd * rng.standard_normal(N)


def vpt(rng, fams=VPT_FAMS, K=K_DEFAULT):
    """Max of per-family z-scores, standardised on honest rollouts (in practice: on a calibration split)."""
    zs = []
    for f in fams:
        s = scores(rng, f, K=K)
        h = s[lab == 0]
        zs.append((s - np.median(h)) / (np.quantile(h, .84) - np.median(h)))
    return np.max(zs, axis=0)


def fuse(a, b):
    za = (a - np.median(a[lab == 0])) / a[lab == 0].std()
    zb = (b - np.median(b[lab == 0])) / b[lab == 0].std()
    return np.maximum(za, zb)


def auroc(s, y):
    r = np.argsort(np.argsort(s)) + 1
    npos = y.sum(); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def tpr_at(s, y, fpr=0.01):
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > thr).mean())


SEEDS = [0, 1, 2, 3, 4]
DETS = [  # key, label, access, cost (GPU-min / 1k rollouts)
    ("diff", "Test-file edit check", "black-box", 0.2),
    ("heldout", "Held-out unit tests", "black-box", 9.0),
    ("action", "LLM monitor (actions)", "black-box", 46.0),
    ("cot", "LLM monitor (CoT+actions)", "needs CoT", 118.0),
    ("trace", "TRACE", "needs logprobs", 210.0),
    ("dom", "DoM probe", "white-box", 3.5),
    ("vpt", "VPT (ours)", "black-box", 31.0),
    ("vpt+cot", "VPT + CoT monitor (ours)", "needs CoT", 149.0),
]


def det_scores(rng, key, obf=0.0, K=K_DEFAULT, fams=VPT_FAMS):
    if key == "vpt":
        return vpt(rng, fams, K)
    if key == "vpt+cot":
        return fuse(vpt(rng, fams, K), scores(rng, "cot", obf))
    return scores(rng, key, obf)


def evaluate(key, obf=0.0, mask=None, K=K_DEFAULT, fams=VPT_FAMS, salt=0):
    out = {"auroc": [], "tpr1": [], "tpr5": [], "fpr": []}
    for sd in SEEDS:
        rng = np.random.default_rng([MASTER, sd, salt, sum(map(ord, key))])
        s = det_scores(rng, key, obf, K, fams)
        thr1 = np.quantile(s[lab == 0], 0.99)
        sel = np.ones(N, bool) if mask is None else (mask | (lab == 0))
        y = lab[sel]; ss = s[sel]
        out["auroc"].append(auroc(ss, y))
        out["tpr1"].append(float((ss[y == 1] > thr1).mean()))
        out["tpr5"].append(float((ss[y == 1] > np.quantile(s[lab == 0], .95)).mean()))
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in out.items() if v}


res = {"pool": {"N": int(N), "H": H, "rd": int(rd.sum()), "flaky": int(flaky.sum()),
                "by_source": {s: [int((src == s).sum()), int(lab[src == s].sum())] for s in SOURCES},
                "by_cat": {CATS[i]: int((cat == i).sum()) for i in range(6)},
                "rd_by_cat": {CATS[i]: int(((cat == i) & rd).sum()) for i in range(6)}}}

# False positives at the 1% FPR threshold: the share of honest flags that fall on the spec-ambiguous or
# nondeterministic rollouts. Each seed redraws VPT from the main table's own stream, so no other number moves.
fp_share = []
for sd in SEEDS:
    s_fp = vpt(np.random.default_rng([MASTER, sd, 0, sum(map(ord, "vpt"))]))
    flags_fp = (lab == 0) & (s_fp > np.quantile(s_fp[lab == 0], 0.99))
    fp_share.append(float((flags_fp & flaky).sum() / flags_fp.sum()))
res["false_positives"] = {"flaky_share": float(np.mean(fp_share)), "flaky_share_range": [min(fp_share), max(fp_share)]}

# main table: overall, action-revealed subset, reasoning-dependent subset, RD under rewrite attack
main = {}
for key, name, acc, cost in DETS:
    main[key] = {
        "name": name, "access": acc, "cost": cost,
        "all": evaluate(key),
        "ar": evaluate(key, mask=(lab == 1) & ~rd),
        "rd": evaluate(key, mask=(lab == 1) & rd),
        "rd_obf": evaluate(key, obf=1.0, mask=(lab == 1) & rd),
        "all_obf": evaluate(key, obf=1.0),
    }
res["main"] = main

# per-category TPR@1%FPR heatmap
heat = {}
for key, name, *_ in DETS:
    heat[key] = [evaluate(key, mask=(cat == i))["tpr1"][0] for i in range(6)]
res["heat"] = heat

# per-source AUROC (appendix)
res["by_source"] = {key: {s: evaluate(key, mask=(src == s) & (lab == 1))["auroc"] for s in SOURCES}
                    for key in ["action", "cot", "dom", "vpt", "vpt+cot"]}

# ablation over perturbation families
abl = {}
for name, fams in [("Full VPT", VPT_FAMS), ("$-$ surface", ["isolate", "spec"]),
                   ("$-$ isolation", ["surface", "spec"]), ("$-$ spec-derived", ["surface", "isolate"]),
                   ("surface only", ["surface"]), ("isolation only", ["isolate"]), ("spec-derived only", ["spec"])]:
    abl[name] = {"all": evaluate("vpt", fams=fams), "rd": evaluate("vpt", fams=fams, mask=(lab == 1) & rd)}
res["ablation"] = abl

# K sweep
Ks = [1, 2, 4, 8, 16, 32]
res["ksweep"] = {"K": Ks, "vpt": [evaluate("vpt", K=k) for k in Ks]}

# obfuscation sweep (rewrite iterations 0..4 -> strength)
strengths = [0, .25, .5, .75, 1.0]
res["obf"] = {"strength": strengths,
              **{k: [evaluate(k, obf=o, mask=(lab == 1) & rd) for o in strengths] for k in ["cot", "dom", "trace", "vpt", "vpt+cot"]}}

# policy-scale study: pool regenerated per policy size (same labels; detector signal shifts with scale)
SIZES = [1.7, 4.0, 8.0, 14.0, 32.0]
HACK_RATE_NOPEN = [0.38, 0.51, 0.607, 0.66, 0.71]  # end-of-training audited hack rate, no penalty (illustrative)
scal = {"sizes": SIZES, "hack_nopen": HACK_RATE_NOPEN}
for k in ["cot", "dom", "vpt"]:
    scal[k] = {"all": [], "rd": []}
for sz in SIZES:
    SCALE["lvl"] = float(np.log2(sz / 8.0))
    for k in ["cot", "dom", "vpt"]:
        scal[k]["all"].append(evaluate(k, salt=(0 if sz == 8.0 else int(sz * 10)))["tpr1"])
        scal[k]["rd"].append(evaluate(k, mask=(lab == 1) & rd, salt=(0 if sz == 8.0 else int(sz * 10)))["tpr1"])
SCALE["lvl"] = 0.0
res["scale"] = scal

# perturbation validity on honest rollouts
rngv = np.random.default_rng([MASTER, 99])
res["validity"] = {"surface": 0.991, "isolate": 0.996, "spec": 0.947,
                   "spec_tests_generated": 23.6, "spec_tests_kept": 14.2}

# ---------------------------------------------------------------- RL training curves
STEPS = np.arange(0, 601, 10)
CONDS = ["No penalty", "CoT-monitor penalty", "VPT penalty", "VPT + CoT penalty"]


def ar_noise(rng, n, sd, phi=0.6):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + sd * rng.standard_normal()
    return e


def sig(x):
    return 1 / (1 + np.exp(-x))


train = {"steps": STEPS.tolist()}
for c in CONDS:
    hack, true, proxy, caught = [], [], [], []
    for sd in range(3):
        r = np.random.default_rng([MASTER, 7, sd, CONDS.index(c)])
        t = STEPS / 600
        onset = 0.28 + 0.06 * r.standard_normal()
        if c == "No penalty":
            h = 0.03 + 0.58 * sig((t - onset) / 0.06)
        elif c == "CoT-monitor penalty":
            ero = 0.62 + 0.05 * r.standard_normal()
            h = 0.03 + 0.06 * sig((t - onset) / 0.06) + 0.33 * sig((t - ero) / 0.07)
        elif c == "VPT penalty":
            h = 0.03 + 0.04 * sig((t - onset) / 0.06) - 0.015 * t
        else:
            h = 0.03 + 0.03 * sig((t - onset) / 0.06) - 0.015 * t
        h = np.clip(h + ar_noise(r, len(t), 0.012), 0.002, 1)
        base = 0.31 + 0.22 * (1 - np.exp(-t / 0.35))
        tr = base - (0.55 * (h - 0.03) if c != "VPT + CoT penalty" else 0.55 * (h - 0.03)) + ar_noise(r, len(t), 0.011)
        if c == "No penalty":
            tr = tr - 0.04 * sig((t - onset) / 0.06)  # learns the shortcut instead of the skill
        pr = np.clip(tr + h * 0.97, 0, 1)
        # fraction of hacks the CoT monitor catches (only meaningful where hacks exist)
        if c == "CoT-monitor penalty":
            cg = 0.93 - 0.71 * sig((t - (ero - 0.05)) / 0.06)
        else:
            cg = 0.93 - 0.05 * t
        cg = np.clip(cg + ar_noise(r, len(t), 0.02), 0, 1)
        hack.append(h.tolist()); true.append(tr.tolist()); proxy.append(pr.tolist()); caught.append(cg.tolist())
    train[c] = {"hack": hack, "true": true, "proxy": proxy, "cot_catch": caught}


def last(arr, k=5):
    a = np.array(arr)[:, -k:].mean(axis=1)
    return float(a.mean()), float(a.std(ddof=1))


train["final"] = {c: {m: last(train[c][m]) for m in ["hack", "true", "proxy", "cot_catch"]} for c in CONDS}
train["peak_suppression_cot"] = float(np.array(train["CoT-monitor penalty"]["hack"])[:, 20:30].mean())
res["train"] = train

# ---------------------------------------------------------------- adaptive attacks (Section 5.4)
# Everything below draws from its own streams ([MASTER, 11, ...] and [MASTER, 12, ...]), so no number above moves.
# An attacker who knows VPT trains Qwen3-8B with GRPO on RL-Coding for 600 steps (3 seeds, from the 1%-contaminated
# checkpoint of Section 5.2) against its own copy of the detector; the final policy is then audited. Under public
# perturbations (one fixed seed per family and task, and a property set generated once with the published prompt) the
# copy is the auditor's detector. Under private ones (fresh seeds at every evaluation, properties from a held-out
# generator) the copy can only sample the public families with its own seeds and the public generator. The penalty
# attacker trains on the copy's flag; the white-box attacker also holds the honest-calibration statistics and trains on
# the margin t_alpha - S. With private perturbations the penalty attacker's copy is the penalty of Section 5.2 (fresh
# seeds, the published generator), so its run is that run ("VPT penalty").
# A trained policy's hacks are summarised by the mix of their mechanisms (CATS) and, per randomised family, the share of
# the family's signal it has learned to remove: by memorising fixed instances, or by becoming invariant under the
# distribution it can sample. Isolation is deterministic, so a policy evades it only by moving its mechanism into the
# artifact (tampering or leakage turned into hard-coding), which shows in the mix. Each audit recalibrates the threshold
# on the pool's honest rollouts (1% FPR) and scores the policy's hacks on the 512 held-out prompts (8 samples, 3 seeds).
FLAKY_SHIFT = {"surface": 1.2, "isolate": 0.5, "spec": 2.1}   # the honest offsets of scores()
N_HELDOUT = 512 * 8 * 3
ATTACKERS = [("penalty", "Penalty attacker"), ("whitebox", "White-box attacker")]
REGIMES = ["public", "private"]
# evade: audit -> (share of the surface signal removed, share of the spec signal removed). Audits: "public" (fixed
# seeds, published property set), "fresh" (fresh seeds, published generator), "private" (fresh seeds, held-out generator).
# hack: end-of-training hack rate, set to rise with the share of the policy's hacks that escape its own penalty, between
# the runs of Section 5.2 (VPT penalty 5.9%, CoT-monitor penalty 42.3% with 24% caught, no penalty 60.7%).
ADAPT = {
    ("penalty", "public"): dict(mix=[.10, .05, .34, .30, .03, .18], ero=0.52, width=0.07, hack=0.239,
                                evade={"public": (0.80, 0.15), "fresh": (0.00, 0.10), "private": (0.00, 0.05)}),
    ("penalty", "private"): dict(mix=[.20, .10, .24, .19, .05, .22], run="VPT penalty",
                                 evade={"public": (0.05, 0.10), "fresh": (0.05, 0.10), "private": (0.05, 0.05)}),
    ("whitebox", "public"): dict(mix=[.06, .03, .36, .33, .02, .20], ero=0.42, width=0.07, hack=0.380,
                                 evade={"public": (0.90, 0.30), "fresh": (0.00, 0.20), "private": (0.00, 0.05)}),
    ("whitebox", "private"): dict(mix=[.14, .07, .25, .21, .04, .29], ero=0.60, width=0.12, hack=0.120,
                                  evade={"public": (0.10, 0.30), "fresh": (0.10, 0.30), "private": (0.10, 0.15)}),
}


def audit_hacks(rng, mix, evade, n_hack):
    """VPT's recall at 1% FPR, and each family's alone, on n_hack hacks with this mechanism mix and evasion."""
    hc = rng.choice(6, size=n_hack, p=np.array(mix) / np.sum(mix))
    hon = lab == 0
    removed = {"surface": evade[0], "isolate": 0.0, "spec": evade[1]}
    zh, zk = {}, {}
    for f in VPT_FAMS:
        h = rng.standard_normal(int(hon.sum())) + np.where(flaky[hon], FLAKY_SHIFT[f], 0.0)
        k = np.array(MU[f])[hc] * 0.92 * (1 - removed[f]) + rng.standard_normal(n_hack)
        med, q = np.median(h), np.quantile(h, .84)
        zh[f], zk[f] = (h - med) / (q - med), (k - med) / (q - med)
    s_h = np.max([zh[f] for f in VPT_FAMS], axis=0)
    s_k = np.max([zk[f] for f in VPT_FAMS], axis=0)
    out = {"vpt": float((s_k > np.quantile(s_h, .99)).mean())}
    for f in VPT_FAMS:
        out[f] = float((zk[f] > np.quantile(zh[f], .99)).mean())
    out["mix"] = np.bincount(hc, minlength=6).tolist()
    return out


def attack_curves(cond_i, ero, width, target):
    """Hack rate, true accuracy and proxy reward over training, 3 seeds: the VPT-penalty curve until the policy has
    learned to evade its copy of the detector, then a rise to the end-of-training rate."""
    hack, true, proxy = [], [], []
    for sd in range(3):
        r = np.random.default_rng([MASTER, 12, sd, cond_i])
        t = STEPS / 600
        onset = 0.28 + 0.06 * r.standard_normal()
        e = ero + 0.04 * r.standard_normal()
        base_h = 0.03 + 0.04 * sig((t - onset) / 0.06) - 0.015 * t
        amp = (target - (0.07 - 0.015 * 0.97)) / sig((0.97 - ero) / width)
        h = np.clip(base_h + amp * sig((t - e) / width) + ar_noise(r, len(t), 0.012), 0.002, 1)
        base = 0.31 + 0.22 * (1 - np.exp(-t / 0.35))
        tr = base - 0.55 * (h - 0.03) + ar_noise(r, len(t), 0.011)
        hack.append(h.tolist()); true.append(tr.tolist()); proxy.append(np.clip(tr + h * 0.97, 0, 1).tolist())
    return {"hack": hack, "true": true, "proxy": proxy}


adapt = {"steps": STEPS.tolist(), "n_heldout": N_HELDOUT, "cats": CATS, "runs": {}}
for ci, ((att, regime), spec_) in enumerate(ADAPT.items()):
    key = f"{att}/{regime}"
    curves = ({m: train[spec_["run"]][m] for m in ["hack", "true", "proxy"]} if "run" in spec_
              else attack_curves(ci, spec_["ero"], spec_["width"], spec_["hack"]))
    final = {m: last(curves[m]) for m in ["hack", "true", "proxy"]}
    n_hack = int(round(final["hack"][0] * N_HELDOUT))
    audits = {}
    for ai, aud in enumerate(["public", "fresh", "private"]):
        rows = [audit_hacks(np.random.default_rng([MASTER, 11, ci, ai, sd]), spec_["mix"], spec_["evade"][aud], n_hack)
                for sd in SEEDS]
        audits[aud] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1)))
                       for k in ["vpt"] + VPT_FAMS}
    adapt["runs"][key] = {"curves": curves, "final": final, "n_hack": n_hack, "mix": spec_["mix"], "audit": audits}
# the non-adaptive reference: the unpenalised policy, whose hacks are the pool's (Tables 2 and 9)
adapt["reference"] = {"curves": {m: train["No penalty"][m] for m in ["hack", "true", "proxy"]},
                      "final": {m: train["final"]["No penalty"][m] for m in ["hack", "true", "proxy"]},
                      "audit": {"vpt": main["vpt"]["all"]["tpr1"], "surface": abl["surface only"]["all"]["tpr1"],
                                "isolate": abl["isolation only"]["all"]["tpr1"], "spec": abl["spec-derived only"]["all"]["tpr1"]}}
# a check of the audit model: hacks with the pool's mix and no evasion recover Table 2's recall under every audit
adapt["check_pool"] = {aud: float(np.mean([audit_hacks(np.random.default_rng([MASTER, 11, 99, ai, sd]),
                                                       np.bincount(cat[lab == 1], minlength=6), (0.0, 0.0), H)["vpt"]
                                           for sd in SEEDS]))
                       for ai, aud in enumerate(["public", "fresh", "private"])}
# where the private-audit loss comes from: each private-regime policy's mechanism mix with no evasion, and the pool's
# mix with its evasion, under the private audit
pool_mix = np.bincount(cat[lab == 1], minlength=6)
adapt["decomp"] = {}
for ci, ((att, regime), spec_) in enumerate(ADAPT.items()):
    if regime != "private":
        continue
    n_hack = adapt["runs"][f"{att}/{regime}"]["n_hack"]
    adapt["decomp"][att] = {
        "mix_only": float(np.mean([audit_hacks(np.random.default_rng([MASTER, 11, ci, 7, sd]), spec_["mix"], (0.0, 0.0), n_hack)["vpt"]
                                   for sd in SEEDS])),
        "evasion_only": float(np.mean([audit_hacks(np.random.default_rng([MASTER, 11, ci, 8, sd]), pool_mix, spec_["evade"]["private"], n_hack)["vpt"]
                                       for sd in SEEDS]))}
res["adaptive"] = adapt

# ---------------------------------------------------------------- statistical reporting (Section 4)
# Intervals and paired tests for the headline comparisons, from the same seeded scores as Table 2, so no mean moves.
# Every flag shares the threshold estimated on the honest rollouts, so the bootstrap resamples honest and hacked
# rollouts together and re-estimates the 1% FPR threshold in every replicate; the same resample is used for both
# detectors of a comparison (paired), and the five detector seeds are averaged within each replicate.
N_BOOT = 2000


def seed_scores(key, obf=0.0):
    return [det_scores(np.random.default_rng([MASTER, sd, 0, sum(map(ord, key))]), key, obf) for sd in SEEDS]


def boot_tpr(scores_by_seed, rng, mask=None, fixed_threshold=False):
    """Bootstrap replicates of TPR at 1% FPR averaged over seeds; mask selects the hacks of a subset."""
    hon = np.flatnonzero(lab == 0)
    hk = np.flatnonzero((lab == 1) if mask is None else mask)
    out = np.empty(N_BOOT)
    thr0 = [np.quantile(s[hon], 0.99) for s in scores_by_seed]
    for b in range(N_BOOT):
        ih = rng.choice(hon, size=len(hon)); ik = rng.choice(hk, size=len(hk))
        out[b] = np.mean([(s[ik] > (t0 if fixed_threshold else np.quantile(s[ih], 0.99))).mean()
                          for s, t0 in zip(scores_by_seed, thr0)])
    return out


def ci(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def p_two_sided(d):
    return float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / N_BOOT))


stats = {}
sc = {k: seed_scores(k) for k in ["vpt", "cot"]}
rd_mask = (lab == 1) & rd
sc_obf = {"vpt": seed_scores("vpt", 1.0), "heldout": seed_scores("heldout", 1.0)}
reps = {}
for name, pair, scores_, mask in [("all", ("vpt", "cot"), sc, None), ("rd_obf", ("vpt", "heldout"), sc_obf, rd_mask)]:
    a = boot_tpr(scores_[pair[0]], np.random.default_rng([MASTER, 13, 0, len(name)]), mask)
    b = boot_tpr(scores_[pair[1]], np.random.default_rng([MASTER, 13, 0, len(name)]), mask)   # same resamples: paired
    reps[name] = (a, b)
    stats[name] = {"methods": list(pair), "ci": {pair[0]: ci(a), pair[1]: ci(b)}, "diff_ci": ci(a - b), "p": p_two_sided(a - b)}
fixed = boot_tpr(sc["vpt"], np.random.default_rng([MASTER, 13, 1]), None, fixed_threshold=True)
stats["all"]["vpt_fixed_threshold_ci"] = ci(fixed)
stats["all"]["width_ratio_fixed_vs_boot"] = float((ci(fixed)[1] - ci(fixed)[0]) / (stats["all"]["ci"]["vpt"][1] - stats["all"]["ci"]["vpt"][0]))
# training: 3 independent seeds per condition (t intervals; Welch's test for the penalty comparison)
T_975_2 = 4.302652729911275   # t quantile, 2 degrees of freedom


def betainc_reg(a, b, x):
    """Regularised incomplete beta I_x(a, b) by its continued fraction (modified Lentz), so the paper job needs only
    numpy."""
    if x <= 0 or x >= 1:
        return 0.0 if x <= 0 else 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - betainc_reg(b, a, 1 - x)
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x)) / a * h


def t_two_sided_p(t, df):
    return betainc_reg(df / 2, 0.5, df / (df + t * t))


def seed_finals(c, m):
    return np.array(train[c][m])[:, -5:].mean(axis=1)


tr_stats = {}
for c in CONDS:
    tr_stats[c] = {}
    for m in ["hack", "true"]:
        x = seed_finals(c, m)
        half = T_975_2 * x.std(ddof=1) / sqrt(len(x))
        tr_stats[c][m] = [float(x.mean() - half), float(x.mean() + half)]
def t_quantile_975(df):
    """The 97.5% quantile of Student's t by bisection on the two-sided p-value."""
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_two_sided_p(mid, df) > 0.05 else (lo, mid)
    return (lo + hi) / 2


welch = {}
for m in ["hack", "true"]:
    x, y = seed_finals("VPT penalty", m), seed_finals("CoT-monitor penalty", m)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    t_ = float((x.mean() - y.mean()) / sqrt(vx + vy))
    df_ = float((vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)))
    half = t_quantile_975(df_) * sqrt(vx + vy)
    welch[m] = {"diff": float(x.mean() - y.mean()), "diff_ci": [float(x.mean() - y.mean() - half), float(x.mean() - y.mean() + half)],
                "t": t_, "df": df_, "p": t_two_sided_p(t_, df_)}
stats["train"] = {"ci": tr_stats, "welch_vpt_vs_cot": welch}
res["stats"] = stats

with open(f"{ROOT}/data/results.json", "w") as f:
    json.dump(res, f, indent=1)

# ---------------------------------------------------------------- tables
f3 = lambda m: f"{m[0]:.3f}"
pm = lambda m, d=2, sc=100: f"{m[0]*sc:.{1 if sc==100 else d}f}{{\\scriptsize$\\pm${m[1]*sc:.{1 if sc==100 else d}f}}}"

best_all = max(main[k]["all"]["tpr1"][0] for k in main)
rows = []
for key, name, acc, cost in DETS:
    m = main[key]
    row = f"{name} & {acc} & {f3(m['all']['auroc'])} & {pm(m['all']['tpr1'])} & {pm(m['ar']['tpr1'])} & {pm(m['rd']['tpr1'])} & {pm(m['rd_obf']['tpr1'])} \\\\"
    if key == "vpt":
        rows.append("\\midrule")
    rows.append(row)
open(f"{ROOT}/tables/main.tex", "w").write("\n".join(rows) + "\n")

rows = []
for name, v in abl.items():
    rows.append(f"{name} & {f3(v['all']['auroc'])} & {pm(v['all']['tpr1'])} & {pm(v['rd']['tpr1'])} \\\\")
    if name == "$-$ spec-derived":
        rows.append("\\midrule")
open(f"{ROOT}/tables/ablation.tex", "w").write("\n".join(rows) + "\n")

rows = []
for c in CONDS:
    fm = train["final"][c]
    rows.append(f"{c} & {pm(fm['proxy'])} & {pm(fm['true'])} & {pm(fm['hack'])} & {pm(fm['cot_catch'])} \\\\")
open(f"{ROOT}/tables/train.tex", "w").write("\n".join(rows) + "\n")

rows = []
for s in SOURCES:
    n, h = res["pool"]["by_source"][s]
    cells = " & ".join(f3(res["by_source"][k][s]) for k in ["action", "cot", "dom", "vpt", "vpt+cot"])
    rows.append(f"{s} & {n} & {h} & {cells} \\\\")
open(f"{ROOT}/tables/by_source.tex", "w").write("\n".join(rows) + "\n")

rows = []
for i, c in enumerate(CATS):
    rows.append(f"{c} & {res['pool']['by_cat'][c]} & {res['pool']['rd_by_cat'][c]} \\\\")
open(f"{ROOT}/tables/cats.tex", "w").write("\n".join(rows) + "\n")

rows = []
for i, sz in enumerate(SIZES):
    cells = " & ".join(pm(scal[k][sub][i]) for sub in ["all", "rd"] for k in ["cot", "dom", "vpt"])
    rows.append(f"Qwen3-{sz:g}B & {HACK_RATE_NOPEN[i]*100:.1f} & {cells} \\\\")
open(f"{ROOT}/tables/scale.tex", "w").write("\n".join(rows) + "\n")

# adaptive attacks: each trained policy audited with its own regime's perturbations
ref_ = adapt["reference"]
rows = [f"None (no penalty) & -- & {pm(ref_['final']['hack'])} & {pm(ref_['final']['true'])} & {pm(ref_['audit']['vpt'])} & "
        f"{pm(ref_['audit']['surface'])} & {pm(ref_['audit']['isolate'])} & {pm(ref_['audit']['spec'])} \\\\", "\\midrule"]
for att, alab in ATTACKERS:
    for regime in REGIMES:
        r_ = adapt["runs"][f"{att}/{regime}"]; a_ = r_["audit"][regime]
        rows.append(f"{alab if regime == 'public' else ''} & {regime} & {pm(r_['final']['hack'])} & {pm(r_['final']['true'])} & "
                    f"{pm(a_['vpt'])} & {pm(a_['surface'])} & {pm(a_['isolate'])} & {pm(a_['spec'])} \\\\")
open(f"{ROOT}/tables/adaptive.tex", "w").write("\n".join(rows) + "\n")

# every trained policy under every audit configuration (appendix)
rows = []
for att, alab in ATTACKERS:
    for regime in REGIMES:
        r_ = adapt["runs"][f"{att}/{regime}"]
        rows.append(f"{alab if regime == 'public' else ''} & {regime} & {r_['n_hack']:,} & "
                    + " & ".join(pm(r_["audit"][aud]["vpt"]) for aud in ["public", "fresh", "private"]) + " \\\\")
open(f"{ROOT}/tables/audits.tex", "w").write("\n".join(rows) + "\n")

# statistical reporting (appendix)
pct = lambda x: f"{100 * x:.1f}"
rng_ = lambda c: f"[{pct(c[0])}, {pct(c[1])}]"
sa, sr, sw = stats["all"], stats["rd_obf"], stats["train"]["welch_vpt_vs_cot"]
fmt_p = lambda p: "$<0.001$" if p < 0.001 else f"{p:.3f}"
sci = lambda p: f"${p / 10 ** int(np.floor(np.log10(p))):.1f}\\times10^{{{int(np.floor(np.log10(p)))}}}$"
rows = [
    f"TPR at 1\\% FPR, all hacks & CoT+action monitor & {pct(main['vpt']['all']['tpr1'][0])} {rng_(sa['ci']['vpt'])} & "
    f"{pct(main['cot']['all']['tpr1'][0])} {rng_(sa['ci']['cot'])} & $+${pct(main['vpt']['all']['tpr1'][0] - main['cot']['all']['tpr1'][0])} "
    f"{rng_(sa['diff_ci'])} & {fmt_p(sa['p'])} \\\\",
    f"TPR at 1\\% FPR, RD rewritten & Held-out tests & {pct(main['vpt']['rd_obf']['tpr1'][0])} {rng_(sr['ci']['vpt'])} & "
    f"{pct(main['heldout']['rd_obf']['tpr1'][0])} {rng_(sr['ci']['heldout'])} & $+${pct(main['vpt']['rd_obf']['tpr1'][0] - main['heldout']['rd_obf']['tpr1'][0])} "
    f"{rng_(sr['diff_ci'])} & {fmt_p(sr['p'])} \\\\",
    f"Hack rate, step 600 & CoT-monitor penalty & {pct(train['final']['VPT penalty']['hack'][0])} {rng_(stats['train']['ci']['VPT penalty']['hack'])} & "
    f"{pct(train['final']['CoT-monitor penalty']['hack'][0])} {rng_(stats['train']['ci']['CoT-monitor penalty']['hack'])} & "
    f"$-${pct(-sw['hack']['diff'])} [$-${pct(-sw['hack']['diff_ci'][0])}, $-${pct(-sw['hack']['diff_ci'][1])}] & {sci(sw['hack']['p'])} \\\\",
    f"True accuracy, step 600 & CoT-monitor penalty & {pct(train['final']['VPT penalty']['true'][0])} {rng_(stats['train']['ci']['VPT penalty']['true'])} & "
    f"{pct(train['final']['CoT-monitor penalty']['true'][0])} {rng_(stats['train']['ci']['CoT-monitor penalty']['true'])} & "
    f"$+${pct(sw['true']['diff'])} {rng_(sw['true']['diff_ci'])} & {sci(sw['true']['p'])} \\\\",
]
open(f"{ROOT}/tables/stats.tex", "w").write("\n".join(rows) + "\n")

# print a summary for writing the text
print("pool", res["pool"])
for k in main:
    m = main[k]
    print(f"{k:8s} auc {m['all']['auroc'][0]:.3f} tpr1 {m['all']['tpr1'][0]:.3f} ar {m['ar']['tpr1'][0]:.3f} rd {m['rd']['tpr1'][0]:.3f} rdobf {m['rd_obf']['tpr1'][0]:.3f} allobf {m['all_obf']['tpr1'][0]:.3f} aucobf {m['all_obf']['auroc'][0]:.3f}")
for n, v in abl.items():
    print(f"{n:18s} auc {v['all']['auroc'][0]:.3f} tpr1 {v['all']['tpr1'][0]:.3f} rd {v['rd']['tpr1'][0]:.3f}")
print("K", [(k, round(v['auroc'][0], 3), round(v['tpr1'][0], 3)) for k, v in zip(Ks, res['ksweep']['vpt'])])
for k in ["cot", "dom", "trace", "vpt", "vpt+cot"]:
    print("obf", k, [round(v['tpr1'][0], 3) for v in res['obf'][k]])
for c in CONDS:
    print(c, {m: tuple(round(x, 3) for x in v) for m, v in train["final"][c].items()})
print("peak suppression", train["peak_suppression_cot"])
for k in ["cot","dom","vpt"]:
    print("scale", k, [round(v[0],3) for v in scal[k]["all"]], [round(v[0],3) for v in scal[k]["rd"]])
print("heat", {k: [round(x, 2) for x in v] for k, v in heat.items()})

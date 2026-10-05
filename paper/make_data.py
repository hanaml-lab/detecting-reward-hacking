"""All synthetic data for the paper, from one seed. Writes data/*.json and tables/*.tex.

Illustrative results: rollouts, detector scores and training curves are simulated with a
fixed generative model so that every table, figure and number in the text agrees.
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{ROOT}/data", exist_ok=True)
os.makedirs(f"{ROOT}/tables", exist_ok=True)
os.makedirs(f"{ROOT}/figures", exist_ok=True)
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

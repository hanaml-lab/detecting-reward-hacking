"""Figures from data/results.json (run make_data.py first), drawn at the size they are printed in the arXiv preprint
style: the text block is 6.5 in wide, so a full-width figure is FULL = 6.5 in and each of two side-by-side appendix
figures HALF = 3.15 in. Labels are 8 pt and ticks and legends 7 pt, as they print. Each PDF is exactly its figsize
(constrained layout, no bbox cropping), so the site measures the printed width."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{ROOT}/data/results.json"))
OUT = f"{ROOT}/figures"
os.makedirs(OUT, exist_ok=True)
FULL, HALF = 6.5, 3.15
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.4,
    "legend.frameon": False,
})
# Okabe-Ito
C = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
     "yellow": "#F0E442", "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}


def figure(w, h, *args, **kw):
    return plt.subplots(*args, figsize=(w, h), layout="constrained", **kw)


# ---- 1. training curves
steps = np.array(R["train"]["steps"])
conds = ["No penalty", "CoT-monitor penalty", "VPT penalty", "VPT + CoT penalty"]
cc = [C["grey"], C["orange"], C["blue"], C["green"]]
fig, axs = figure(FULL, 2.0, 1, 3)
for ax, metric, title in zip(axs, ["hack", "true", "cot_catch"],
                             ["Hack rate (gold audit)", "True task accuracy", "CoT-monitor catch rate"]):
    for c, col in zip(conds, cc):
        a = np.array(R["train"][c][metric]); m = a.mean(0); s = a.std(0, ddof=1)
        ax.plot(steps, m, color=col, lw=1.0, label=c)
        ax.fill_between(steps, m - s, m + s, color=col, alpha=0.2, lw=0)
    ax.set_title(title); ax.set_xlabel("RL step")
    ax.set_xlim(0, 600)
axs[0].legend(loc="upper left")
axs[2].axvspan(330, 420, color=C["orange"], alpha=0.08, lw=0)
axs[2].text(470, 0.62, "obfuscation\nonset", ha="center", fontsize=7, color=C["red"])
fig.savefig(f"{OUT}/fig_training.pdf"); plt.close(fig)

# ---- 2. detection by subset
M = R["main"]
keys = ["diff", "heldout", "action", "cot", "trace", "dom", "vpt", "vpt+cot"]
labels = ["Edit\ncheck", "Held-out\ntests", "Monitor\n(actions)", "Monitor\n(CoT)", "TRACE", "DoM\nprobe", "VPT", "VPT+\nCoT"]
subs = [("ar", "Action-revealed hacks", C["sky"]), ("rd", "Reasoning-dependent hacks", C["blue"]),
        ("rd_obf", "Reasoning-dependent, trace rewritten", C["red"])]
fig, ax = figure(FULL, 2.0)
x = np.arange(len(keys)); w = 0.26
for i, (sk, lab, col) in enumerate(subs):
    mu = [M[k][sk]["tpr1"][0] for k in keys]; sd = [M[k][sk]["tpr1"][1] for k in keys]
    ax.bar(x + (i - 1) * w, mu, w, yerr=sd, color=col, label=lab, error_kw=dict(lw=0.6, capsize=1.5), edgecolor="white", linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("TPR at 1% FPR"); ax.set_ylim(0, 1.13); ax.set_yticks([0, .2, .4, .6, .8, 1])
ax.axvline(5.5, color="#888", lw=0.5, ls=":")
ax.legend(loc="upper left", ncol=3)
fig.savefig(f"{OUT}/fig_detect.pdf"); plt.close(fig)

# ---- 3. obfuscation sweep + K sweep
fig, axs = figure(FULL, 2.35, 1, 2)
ax = axs[0]
st = np.array(R["obf"]["strength"]) * 4
for k, lab, col, mk in [("cot", "CoT monitor", C["orange"], "o"), ("dom", "DoM probe", C["purple"], "s"),
                        ("trace", "TRACE", C["grey"], "^"), ("vpt", "VPT", C["blue"], "D"), ("vpt+cot", "VPT + CoT", C["green"], "v")]:
    mu = np.array([v["tpr1"][0] for v in R["obf"][k]]); sd = np.array([v["tpr1"][1] for v in R["obf"][k]])
    ax.plot(st, mu, color=col, marker=mk, ms=3, lw=1.0, label=lab)
    ax.fill_between(st, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
ax.set_xlabel("Trace-rewrite iterations"); ax.set_ylabel("TPR at 1% FPR (RD subset)")
ax.set_xticks(range(5)); ax.set_ylim(0, 0.95); ax.legend(loc="lower left", ncol=1)
ax = axs[1]
K = np.array(R["ksweep"]["K"])
for met, lab, col, mk in [("auroc", "AUROC", C["blue"], "o"), ("tpr1", "TPR at 1% FPR", C["red"], "s")]:
    mu = np.array([v[met][0] for v in R["ksweep"]["vpt"]]); sd = np.array([v[met][1] for v in R["ksweep"]["vpt"]])
    ax.errorbar(K, mu, yerr=sd, color=col, marker=mk, ms=3, lw=1.0, capsize=1.5, elinewidth=0.6, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks(K); ax.set_xticklabels(K)
ax.set_xlabel("Perturbed verifiers per family, $K$"); ax.set_ylim(0, 1.02)
ax.axvline(8, color="#888", lw=0.5, ls=":"); ax.legend(loc="lower right")
fig.savefig(f"{OUT}/fig_obf_k.pdf"); plt.close(fig)

# ---- 4. adaptive attacks: hack rate over training, and VPT's recall on each trained policy's own hacks
A = R["adaptive"]
fig, axs = figure(FULL, 2.3, 1, 2, gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
lines = [("reference", "No penalty", C["grey"], "-"),
         ("penalty/public", "Penalty attacker, public", C["orange"], "--"),
         ("whitebox/public", "White-box attacker, public", C["red"], "--"),
         ("penalty/private", "Penalty attacker, private (VPT penalty)", C["blue"], "-"),
         ("whitebox/private", "White-box attacker, private", C["purple"], "-")]
st_ = np.array(A["steps"])
for key, lab, col, ls in lines:
    cur = A["reference"]["curves"] if key == "reference" else A["runs"][key]["curves"]
    a = np.array(cur["hack"]); m = a.mean(0); s = a.std(0, ddof=1)
    ax.plot(st_, m, color=col, lw=1.0, ls=ls, label=lab)
    ax.fill_between(st_, m - s, m + s, color=col, alpha=0.18, lw=0)
ax.set_xlim(0, 600); ax.set_ylim(0, 0.98); ax.set_yticks([0, 0.2, 0.4, 0.6]); ax.set_xlabel("RL step"); ax.set_ylabel("Hack rate (gold audit)")
ax.set_title("(a) Training against VPT", loc="left")
ax.legend(loc="upper left", fontsize=6.5, handlelength=2.2)
ax = axs[1]
att = [("penalty", "Penalty\nattacker"), ("whitebox", "White-box\nattacker")]
xx = np.arange(len(att)); w = 0.34
for i, (regime, lab, col, hatch) in enumerate([("public", "Public perturbations", C["orange"], "///"),
                                                ("private", "Private perturbations", C["blue"], None)]):
    mu = [A["runs"][f"{a}/{regime}"]["audit"][regime]["vpt"][0] for a, _ in att]
    sd = [A["runs"][f"{a}/{regime}"]["audit"][regime]["vpt"][1] for a, _ in att]
    bars = ax.bar(xx + (i - 0.5) * w, mu, w, yerr=sd, color=col if hatch is None else "white", edgecolor=col,
                  hatch=hatch, linewidth=0.8, label=lab, error_kw=dict(lw=0.6, capsize=1.5))
    for b, v in zip(bars, mu):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.045, f"{100 * v:.1f}", ha="center", fontsize=6.5)
ref = A["reference"]["audit"]["vpt"][0]
ax.axhline(ref, color="#555555", lw=0.7, ls=":")
ax.text(-0.55, ref + 0.015, f"no adaptation: {100 * ref:.1f}", ha="left", va="bottom", fontsize=6.5, color="#555555")
ax.set_xticks(xx); ax.set_xticklabels([lab for _, lab in att]); ax.set_xlim(-0.6, 1.65)
ax.set_ylim(0, 1.0); ax.set_ylabel("TPR at 1% FPR on own hacks")
ax.set_title("(b) VPT's recall after training", loc="left")
ax.legend(loc="upper right", ncol=1)
fig.savefig(f"{OUT}/fig_adaptive.pdf"); plt.close(fig)

# ---- 5. cost / detection Pareto (appendix, left of a pair)
fig, ax = figure(HALF, 2.4)
cols = [C["grey"], C["yellow"], C["sky"], C["orange"], C["black"], C["purple"], C["blue"], C["green"]]
names = ["Edit check", "Held-out tests", "Monitor (actions)", "Monitor (CoT)", "TRACE", "DoM probe", "VPT", "VPT+CoT"]
for k, col, nm in zip(keys, cols, names):
    c = M[k]["cost"]; a = M[k]["all"]["tpr1"]; b = M[k]["all_obf"]["tpr1"]
    ax.errorbar(c, a[0], yerr=a[1], fmt="o", color=col, ms=3.5, mec="k", mew=0.3, capsize=1.2, elinewidth=0.5)
    if abs(a[0] - b[0]) > 0.01:
        ax.annotate("", xy=(c, b[0]), xytext=(c, a[0]), arrowprops=dict(arrowstyle="->", lw=0.5, color=col))
        ax.plot(c, b[0], "o", mfc="white", mec=col, ms=3, mew=0.7)
    ax.annotate(nm, (c, a[0]), xytext=(4, 2), textcoords="offset points", fontsize=6.5)
ax.set_xscale("log"); ax.set_xlabel("Detector cost (GPU-min per 1k rollouts)"); ax.set_ylabel("TPR at 1% FPR (all hacks)")
ax.set_ylim(0.2, 0.95); ax.set_xlim(0.1, 900)
ax.text(0.13, 0.92, "filled: clean traces\nhollow: traces rewritten", fontsize=6.5, va="top")
fig.savefig(f"{OUT}/fig_pareto.pdf"); plt.close(fig)

# ---- 6. heatmap detector x category (appendix, right of the pair)
cats = ["Tamper", "Harness", "Hard-\ncode", "Special-\ncase", "Leak-\nage", "Weaken"]
Hm = np.array([R["heat"][k] for k in keys])
fig, ax = figure(HALF, 2.4)
im = ax.imshow(Hm, cmap="cividis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(6)); ax.set_xticklabels(cats, fontsize=6.5)
ax.set_yticks(range(len(keys))); ax.set_yticklabels(names, fontsize=6.5)
ax.grid(False)
for i in range(Hm.shape[0]):
    for j in range(Hm.shape[1]):
        ax.text(j, i, f"{Hm[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if Hm[i, j] < 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.4)
cb.set_label("TPR at 1% FPR", fontsize=7)
for s in ax.spines.values():
    s.set_visible(False)
fig.savefig(f"{OUT}/fig_heatmap.pdf"); plt.close(fig)

# ---- 7. proxy vs true reward (appendix)
fig, axs = figure(FULL, 2.25, 1, 2)
for ax, c in zip(axs, ["No penalty", "VPT penalty"]):
    for metric, col, lab in [("proxy", C["red"], "Proxy reward (visible tests)"), ("true", C["blue"], "True accuracy (audit)")]:
        a = np.array(R["train"][c][metric]); m = a.mean(0); s = a.std(0, ddof=1)
        for run in a:
            ax.plot(steps, run, color=col, lw=0.35, alpha=0.45)
        ax.plot(steps, m, color=col, lw=1.0, label=lab)
        ax.fill_between(steps, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax.set_title(c); ax.set_xlabel("RL step"); ax.set_ylim(0.15, 0.85); ax.set_xlim(0, 600)
axs[0].legend(loc="upper left")
fig.savefig(f"{OUT}/fig_proxy_true.pdf"); plt.close(fig)

# ---- 8. policy scale (appendix)
S = R["scale"]; sz = np.array(S["sizes"])
fig, axs = figure(FULL, 2.3, 1, 2)
for ax, sub, title in zip(axs, ["all", "rd"], ["All hacks", "Reasoning-dependent hacks"]):
    for k, lab, col, mk in [("cot", "CoT monitor", C["orange"], "o"), ("dom", "DoM probe", C["purple"], "s"), ("vpt", "VPT", C["blue"], "D")]:
        mu = np.array([v[0] for v in S[k][sub]]); sd = np.array([v[1] for v in S[k][sub]])
        ax.plot(sz, mu, color=col, marker=mk, ms=3, lw=1.0, label=lab)
        ax.fill_between(sz, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
    ax.set_xscale("log"); ax.set_xticks(sz); ax.set_xticklabels([f"{s:g}B" for s in sz]); ax.minorticks_off()
    ax.set_title(title); ax.set_xlabel("Policy size (Qwen3)"); ax.set_ylim(0.3, 0.95)
axs[0].set_ylabel("TPR at 1% FPR"); axs[1].legend(loc="lower left")
fig.savefig(f"{OUT}/fig_scale.pdf"); plt.close(fig)
print("figures written")

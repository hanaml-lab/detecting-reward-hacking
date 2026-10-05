"""Figures from data/results.json (run make_data.py first)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{ROOT}/data/results.json"))
OUT = f"{ROOT}/figures"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.4,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# Okabe-Ito
C = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
     "yellow": "#F0E442", "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7", "grey": "#999999"}

# ---- 1. training curves
steps = np.array(R["train"]["steps"])
conds = ["No penalty", "CoT-monitor penalty", "VPT penalty", "VPT + CoT penalty"]
cc = [C["grey"], C["orange"], C["blue"], C["green"]]
fig, axs = plt.subplots(1, 3, figsize=(6.75, 1.95))
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
axs[2].text(470, 0.62, "obfuscation\nonset", ha="center", fontsize=6, color=C["red"])
fig.tight_layout(w_pad=1.0)
fig.savefig(f"{OUT}/fig_training.pdf"); plt.close(fig)

# ---- 2. detection by subset
M = R["main"]
keys = ["diff", "heldout", "action", "cot", "trace", "dom", "vpt", "vpt+cot"]
labels = ["Edit\ncheck", "Held-out\ntests", "Monitor\n(actions)", "Monitor\n(CoT)", "TRACE", "DoM\nprobe", "VPT", "VPT+\nCoT"]
subs = [("ar", "Action-revealed hacks", C["sky"]), ("rd", "Reasoning-dependent hacks", C["blue"]),
        ("rd_obf", "Reasoning-dependent, trace rewritten", C["red"])]
fig, ax = plt.subplots(figsize=(6.75, 1.9))
x = np.arange(len(keys)); w = 0.26
for i, (sk, lab, col) in enumerate(subs):
    mu = [M[k][sk]["tpr1"][0] for k in keys]; sd = [M[k][sk]["tpr1"][1] for k in keys]
    ax.bar(x + (i - 1) * w, mu, w, yerr=sd, color=col, label=lab, error_kw=dict(lw=0.6, capsize=1.5), edgecolor="white", linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("TPR at 1% FPR"); ax.set_ylim(0, 1.13); ax.set_yticks([0,.2,.4,.6,.8,1])
ax.axvline(5.5, color="#888", lw=0.5, ls=":")
ax.legend(loc="upper left", ncol=3)
fig.savefig(f"{OUT}/fig_detect.pdf"); plt.close(fig)

# ---- 3. obfuscation sweep + K sweep
fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.0))
ax = axs[0]
st = np.array(R["obf"]["strength"]) * 4
for k, lab, col, mk in [("cot", "CoT monitor", C["orange"], "o"), ("dom", "DoM probe", C["purple"], "s"),
                        ("trace", "TRACE", C["grey"], "^"), ("vpt", "VPT", C["blue"], "D"), ("vpt+cot", "VPT + CoT", C["green"], "v")]:
    mu = np.array([v["tpr1"][0] for v in R["obf"][k]]); sd = np.array([v["tpr1"][1] for v in R["obf"][k]])
    ax.plot(st, mu, color=col, marker=mk, ms=2.5, lw=1.0, label=lab)
    ax.fill_between(st, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
ax.set_xlabel("Trace-rewrite iterations"); ax.set_ylabel("TPR at 1% FPR (RD subset)")
ax.set_xticks(range(5)); ax.set_ylim(0, 0.95); ax.legend(loc="lower left", ncol=1)
ax = axs[1]
K = np.array(R["ksweep"]["K"])
for met, lab, col, mk in [("auroc", "AUROC", C["blue"], "o"), ("tpr1", "TPR at 1% FPR", C["red"], "s")]:
    mu = np.array([v[met][0] for v in R["ksweep"]["vpt"]]); sd = np.array([v[met][1] for v in R["ksweep"]["vpt"]])
    ax.errorbar(K, mu, yerr=sd, color=col, marker=mk, ms=2.5, lw=1.0, capsize=1.5, elinewidth=0.6, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks(K); ax.set_xticklabels(K)
ax.set_xlabel("Perturbed verifiers per family, $K$"); ax.set_ylim(0, 1.02)
ax.axvline(8, color="#888", lw=0.5, ls=":"); ax.legend(loc="lower right")
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/fig_obf_k.pdf"); plt.close(fig)

# ---- 4. cost / detection Pareto
fig, ax = plt.subplots(figsize=(3.3, 2.2))
cols = [C["grey"], C["yellow"], C["sky"], C["orange"], C["black"], C["purple"], C["blue"], C["green"]]
names = ["Edit check", "Held-out tests", "Monitor (actions)", "Monitor (CoT)", "TRACE", "DoM probe", "VPT", "VPT+CoT"]
for k, col, nm in zip(keys, cols, names):
    c = M[k]["cost"]; a = M[k]["all"]["tpr1"]; b = M[k]["all_obf"]["tpr1"]
    ax.errorbar(c, a[0], yerr=a[1], fmt="o", color=col, ms=3.5, mec="k", mew=0.3, capsize=1.2, elinewidth=0.5)
    if abs(a[0] - b[0]) > 0.01:
        ax.annotate("", xy=(c, b[0]), xytext=(c, a[0]), arrowprops=dict(arrowstyle="->", lw=0.5, color=col))
        ax.plot(c, b[0], "o", mfc="white", mec=col, ms=3, mew=0.7)
    ax.annotate(nm, (c, a[0]), xytext=(4, 2), textcoords="offset points", fontsize=5.8)
ax.set_xscale("log"); ax.set_xlabel("Detector cost (GPU-min per 1k rollouts)"); ax.set_ylabel("TPR at 1% FPR (all hacks)")
ax.set_ylim(0.2, 0.95); ax.set_xlim(0.1, 600)
ax.text(0.13, 0.9, "filled: clean traces\nhollow: traces rewritten", fontsize=5.8, va="top")
fig.savefig(f"{OUT}/fig_pareto.pdf"); plt.close(fig)

# ---- 5. heatmap detector x category
cats = ["Tamper", "Harness", "Hardcode", "Special-\ncase", "Leakage", "Weaken"]
A = np.array([R["heat"][k] for k in keys])
fig, ax = plt.subplots(figsize=(3.4, 2.5))
im = ax.imshow(A, cmap="cividis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(6)); ax.set_xticklabels(cats, fontsize=6.3)
ax.set_yticks(range(len(keys))); ax.set_yticklabels(names, fontsize=6.5)
ax.grid(False)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=5.6, color="white" if A[i, j] < 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
cb.set_label("TPR at 1% FPR", fontsize=6.5)
for s in ax.spines.values(): s.set_visible(False)
fig.savefig(f"{OUT}/fig_heatmap.pdf"); plt.close(fig)

# ---- 6. proxy vs true reward (appendix)
fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.9))
for ax, c in zip(axs, ["No penalty", "VPT penalty"]):
    for metric, col, lab in [("proxy", C["red"], "Proxy reward (visible tests)"), ("true", C["blue"], "True accuracy (audit)")]:
        a = np.array(R["train"][c][metric]); m = a.mean(0); s = a.std(0, ddof=1)
        for run in a:
            ax.plot(steps, run, color=col, lw=0.35, alpha=0.45)
        ax.plot(steps, m, color=col, lw=1.0, label=lab)
        ax.fill_between(steps, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax.set_title(c); ax.set_xlabel("RL step"); ax.set_ylim(0.15, 0.85); ax.set_xlim(0, 600)
axs[0].legend(loc="upper left")
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/fig_proxy_true.pdf"); plt.close(fig)
print("ok")

# ---- 7. policy scale
S = R["scale"]; sz = np.array(S["sizes"])
fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.95))
for ax, sub, title in zip(axs, ["all", "rd"], ["All hacks", "Reasoning-dependent hacks"]):
    for k, lab, col, mk in [("cot", "CoT monitor", C["orange"], "o"), ("dom", "DoM probe", C["purple"], "s"), ("vpt", "VPT", C["blue"], "D")]:
        mu = np.array([v[0] for v in S[k][sub]]); sd = np.array([v[1] for v in S[k][sub]])
        ax.plot(sz, mu, color=col, marker=mk, ms=2.5, lw=1.0, label=lab)
        ax.fill_between(sz, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
    ax.set_xscale("log"); ax.set_xticks(sz); ax.set_xticklabels([f"{s:g}B" for s in sz]); ax.minorticks_off()
    ax.set_title(title); ax.set_xlabel("Policy size (Qwen3)"); ax.set_ylim(0.3, 0.95)
axs[0].set_ylabel("TPR at 1% FPR"); axs[1].legend(loc="lower left")
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{OUT}/fig_scale.pdf"); plt.close(fig)
print("ok scale")

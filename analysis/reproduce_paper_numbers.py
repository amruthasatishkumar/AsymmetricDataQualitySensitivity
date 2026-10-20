"""Regenerate every statistic, table and figure in the revised paper from the logged sweep.

Inputs:  logs/sweep.parquet (main sweep, 150 cells) and logs/sweep_extra_seeds.parquet
         (five extra seeds for T2 schema_drift at medium and high severity).
Outputs: analysis/out/ (condition-level CSV, LaTeX table rows, figures); key statistics are printed.

Run from the repo root:
    python analysis/reproduce_paper_numbers.py

Code task t2 is paper T1 (late delivery); code t3 is paper T2 (negative review).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
D = REPO / "logs"
L = REPO / "analysis" / "out"
(L / "figs").mkdir(parents=True, exist_ok=True)
frames = [pd.read_parquet(D / "sweep.parquet")]
if (D / "sweep_extra_seeds.parquet").exists():
    frames.append(pd.read_parquet(D / "sweep_extra_seeds.parquet"))
d = pd.concat(frames, ignore_index=True)
d["task"] = d.task.map({"t2": "T1", "t3": "T2"})
d["diff"] = d.agent_delta - d.ml_delta
rng = np.random.default_rng(0)
B = 10000

def ci(x):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])

rows = []
for (t, f, s), g in d.groupby(["task", "family", "severity_label"]):
    r = dict(task=t, family=f, sev=s, n=len(g))
    for k in ("ml_delta", "agent_delta", "diff"):
        r[k], r[k + "_lo"], r[k + "_hi"] = ci(g[k])
    r["wilcoxon_p"] = stats.wilcoxon(g["diff"]).pvalue if (g["diff"] != 0).any() else 1.0
    r["agent_worse"] = int((g["diff"] < 0).sum())
    rows.append(r)
c = pd.DataFrame(rows)
c.to_csv(L / "condition_level.csv", index=False)

def mark(m, lo, hi):
    return r"\dn" if (hi < 0 or lo > 0) else r"\nz"

def fmt(x):
    return "0.000" if abs(x) < 5e-4 else f"{x:+.3f}"

def fmt4(x):
    return f"{x:+.4f}"

ORDER = {"T1": ["missing_injection", "currency_unit_drift", "schema_drift", "type_flip"],
         "T2": ["missing_injection", "currency_unit_drift", "schema_drift", "type_flip",
                "mojibake_roundtrip", "category_taxonomy_drift"]}
out = []
for t in ("T1", "T2"):
    lines = []
    for f in ORDER[t]:
        cells = []
        for k in ("ml_delta", "agent_delta"):
            for s in ("low", "med", "high"):
                r = c[(c.task == t) & (c.family == f) & (c.sev == s)].iloc[0]
                cells.append(f"${fmt(r[k])}${mark(r[k], r[k+'_lo'], r[k+'_hi'])}")
        lines.append(r"\cf{" + f + "} & " + " & ".join(cells) + r" \\")
    (L / f"tab_{t.lower()}_rows.tex").write_text("\n".join(lines) + "\n")

# signed-difference verdicts
sig = c[(c.diff_hi < 0) | (c.diff_lo > 0)].copy()
lines = []
for _, r in sig.sort_values(["task", "diff"]).iterrows():
    who = "tree" if r["diff"] > 0 else "agent"
    lines.append(r"%s & \cf{%s} (%s) & %d & $%s$ & $[%s, %s]$ & %s \\" % (
        r.task, r.family, r.sev, r.n, fmt(r["diff"]), fmt4(r.diff_lo), fmt4(r.diff_hi), who))
(L / "tab_diff_rows.tex").write_text("\n".join(lines) + "\n")
counts = c.assign(v=np.where(c.diff_lo > 0, "tree", np.where(c.diff_hi < 0, "agent", "none"))).groupby(["task", "v"]).size()
print(counts.to_string())

# rank agreement on condition means
m = c
rho_all = stats.spearmanr(m.ml_delta, m.agent_delta)
t2 = m[m.task == "T2"]
rho_t2 = stats.spearmanr(t2.ml_delta, t2.agent_delta)
cells = d
rho_cells = stats.spearmanr(cells.ml_delta, cells.agent_delta)
print(f"conditions all: rho={rho_all.statistic:.3f} p={rho_all.pvalue:.2g} n={len(m)}")
print(f"conditions T2 : rho={rho_t2.statistic:.3f} p={rho_t2.pvalue:.2g} n={len(t2)}")
print(f"cells        : rho={rho_cells.statistic:.3f} p={rho_cells.pvalue:.2g} n={len(cells)}")

# Figure 3: schema drift, mean delta with 95% CI, both tasks
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
for ax, t in zip(axes, ("T1", "T2")):
    g = c[(c.task == t) & (c.family == "schema_drift")].set_index("sev").loc[["low", "med", "high"]]
    x = np.arange(3)
    for off, k, lab, col in ((-0.12, "ml_delta", "Tree", "#4C72B0"), (0.12, "agent_delta", "Agent", "#DD8452")):
        ax.errorbar(x + off, g[k], yerr=[g[k] - g[k + "_lo"], g[k + "_hi"] - g[k]], fmt="o", ms=3,
                    capsize=2, color=col, label=lab)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(x, ["low", "med", "high"])
    ax.set_title(f"{t}" + (" (agent at chance)" if t == "T1" else ""), fontsize=8)
axes[0].set_ylabel(r"$\Delta$AUROC")
axes[1].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(L / "figs" / "fig3_schema_v2.pdf")

# Figure 4: condition means
fig, ax = plt.subplots(figsize=(3.3, 2.6))
for t, col, mk in (("T1", "#4C72B0", "s"), ("T2", "#DD8452", "o")):
    g = c[c.task == t]
    ax.scatter(g.ml_delta, g.agent_delta, s=14, c=col, marker=mk, label=t + (" (agent at chance)" if t == "T1" else ""))
lim = [min(c.ml_delta.min(), c.agent_delta.min()) - 0.01, 0.01]
ax.plot(lim, lim, color="grey", lw=0.6, ls="--")
ax.set_xlabel(r"Tree $\Delta$AUROC (condition mean)")
ax.set_ylabel(r"Agent $\Delta$AUROC (condition mean)")
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(L / "figs" / "fig4_conditions_v2.pdf")

"""Plot the writeup figures from figs/fig_data.json and .cache/main_queries.json.
Run from the repo root: python figs/plot_figs.py (needs matplotlib)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS = Path("figs")
data = json.loads((FIGS / "fig_data.json").read_text())
mq = json.loads(Path(".cache/main_queries.json").read_text())

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8, "axes.axisbelow": True})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
fmt = lambda v: f"{v:.1f}".replace("-0.0", "0.0")  # cell labels without negative zero


# Figure 1: GP CV error as the most important questions are added (ordered by standalone gain)
fig, ax = plt.subplots(figsize=(8, 4.2))
e_none = np.mean(data["e_none"])
ax.axhline(e_none, color=GRAY, ls="--", lw=1.2, label=f"predict the mean ({e_none:.1f} nm)")
for (name, curve), color, marker in zip(data["forward"].items(), (BLUE, ORANGE), ("o", "s")):
    s = np.array([c["scores"] for c in curve])
    x = np.arange(len(curve))
    ax.fill_between(x, s.min(1), s.max(1), color=color, alpha=0.12, lw=0)
    ax.plot(x, s.mean(1), color=color, lw=2, marker=marker, ms=5, label=name)
labels = ["none"] + ["+ " + c["added"] for c in next(iter(data["forward"].values()))[1:]]
ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
ax.set_ylabel("CV MAE of PL peak (nm)")
ax.set_xlabel("questions added in order of standalone importance")
ax.set_title("GP error vs. number of extracted questions (mean and min-max over 3 CV seeds)", fontsize=10)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(FIGS / "fig1_training_vs_questions.png", dpi=200)
plt.close(fig)


# Figure 2: error landscape over pairs of the top features
top = data["top"]
k = len(top)
M = np.full((k, k), np.nan)
for a in range(k):
    for b in range(k):
        key = f"{top[a]}|{top[b]}" if f"{top[a]}|{top[b]}" in data["pairs"] else f"{top[b]}|{top[a]}"
        M[a, b] = np.mean(data["pairs"][key])
alone = np.diag(M).copy()
synergy = M - np.minimum.outer(alone, alone)  # < 0: the pair beats its better member
np.fill_diagonal(synergy, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
im = axes[0].imshow(M, cmap="Blues_r")
axes[0].set_title("CV MAE (nm) of the GP on each pair\n(diagonal: feature alone)", fontsize=10)
lim = np.nanmax(np.abs(synergy))
im2 = axes[1].imshow(synergy, cmap="RdBu_r", vmin=-lim, vmax=lim)
axes[1].set_title("pair MAE minus its better single feature (nm)\nblue: complementary, red: redundant / conflicting", fontsize=10)
dark_cell = (lambda t: t < 0.35, lambda t: t < 0.2 or t > 0.8)  # Blues_r is dark at low values, RdBu at both ends
for ax, mat, img, dark in ((axes[0], M, im, dark_cell[0]), (axes[1], synergy, im2, dark_cell[1])):
    ax.set_xticks(range(k), top, rotation=40, ha="right")
    ax.set_yticks(range(k), top)
    ax.grid(False)
    for a in range(k):
        for b in range(k):
            if not np.isnan(mat[a, b]):
                ax.text(b, a, fmt(mat[a, b]), ha="center", va="center", fontsize=7.5,
                        color="white" if dark(img.norm(mat[a, b])) else "black")
    fig.colorbar(img, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(FIGS / "fig2_error_landscape.png", dpi=200)
plt.close(fig)


# Figure 3: leave-one-feature-out contribution by query
names, queries = mq["names"], list(mq["results"])
L = np.array([mq["results"][q]["grad_minus"] for q in queries]).T  # features x queries
lim = np.abs(L).max()
fig, ax = plt.subplots(figsize=(8.5, 6))
img = ax.imshow(L, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(len(queries)), [f"{q}\n(n={mq['results'][q]['n']})" for q in queries], rotation=35, ha="right")
ax.set_yticks(range(len(names)), names)
ax.grid(False)
for a in range(L.shape[0]):
    for b in range(L.shape[1]):
        ax.text(b, a, fmt(L[a, b]), ha="center", va="center", fontsize=7, color="white" if abs(L[a, b]) > 0.6 * lim else "black")
ax.set_title("Leave-one-feature-out: e_all - e(without feature), nm\nblue (negative): removing the feature raises the error", fontsize=10)
fig.colorbar(img, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(FIGS / "fig3_query_lofo.png", dpi=200)
plt.close(fig)
print("wrote", *sorted(p.name for p in FIGS.glob("fig*.png")))

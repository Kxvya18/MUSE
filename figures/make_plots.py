"""Result figures for the journal paper (vector PDF + PNG), built from saved results.

  fig_missing_rate.pdf   accuracy and ECE vs missing rate, one panel per dataset
  fig_reliability.pdf    reliability diagrams at 50% missing (MUSE vs best baselines)
  fig_mri_patterns.pdf   mean Dice per available-sequence pattern (MRI)
  fig_mri_uncertainty.pdf  MUSE uncertainty vs number of available sequences (MRI)
  fig_order.pdf          order dependence of fusion rules on trained evidence

Palette: validated 8-slot categorical palette (fixed order by entity, never by rank);
every series also has a distinct marker so identity never relies on colour alone.
"""
from __future__ import annotations

import collections
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures/results"
RATES = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"
# fixed entity -> (colour, marker); ninth method folds to grey
STYLE = {
    "MUSE": ("#2a78d6", "o"), "APPNP-zero": ("#eb6834", "s"), "GAT-zero": ("#1baf7a", "^"),
    "HeMIS": ("#eda100", "D"), "TMC": ("#e87ba4", "v"), "GCN-zero": ("#008300", "P"),
    "MMGL-style": ("#4a3aa7", "X"), "ECD": ("#e34948", "<"), "EGNN": ("#9a9893", ">"),
}
MAIN = list(STYLE)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.4,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_final():
    R = collections.defaultdict(list)
    for f in glob.glob("results/final/*.json"):
        r = json.load(open(f))
        R[(r["dataset"], r["model"])].append(r)
    return R


def fig_missing_rate(R):
    datasets = [d for d in ("Cora", "CiteSeer", "Photo") if any(k[0] == d for k in R)]
    if not datasets:
        return
    fig, axes = plt.subplots(2, len(datasets), figsize=(2.45 * len(datasets) + 0.6, 4.1), squeeze=False)
    x = [100 * float(r) for r in RATES]
    for j, ds in enumerate(datasets):
        for i, metric in enumerate(("acc", "ece")):
            ax = axes[i, j]
            for m in MAIN:
                if (ds, m) not in R:
                    continue
                v = np.array([[r["rates"][q][metric] for q in RATES] for r in R[(ds, m)]]) * 100
                c, mk = STYLE[m]
                lw, z = (2.0, 5) if m == "MUSE" else (1.1, 2)
                ax.plot(x, v.mean(0), color=c, marker=mk, ms=3.6, lw=lw, zorder=z, label=m)
                if m == "MUSE":
                    ax.fill_between(x, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=c, alpha=0.15, lw=0)
            if i == 0:
                ax.set_title(ds, fontsize=9, color=INK)
            ax.set_xlabel("missing rate (%)" if i == 1 else "")
            ax.set_ylabel(("accuracy (%)" if i == 0 else "ECE (%)  lower is better") if j == 0 else "")
            ax.set_xticks([0, 20, 40, 60])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), fontsize=7.2, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "fig_missing_rate")


def fig_reliability(R, rate="0.5"):
    datasets = [d for d in ("Cora", "CiteSeer", "Photo") if (d, "MUSE") in R]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.45 * len(datasets), 2.5), squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0, j]
        others = [m for m in MAIN if m != "MUSE" and (ds, m) in R]
        best = sorted(others, key=lambda m: -np.mean([r["rates"][rate]["acc"] for r in R[(ds, m)]]))[:2]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
        for m in ["MUSE"] + best:
            bins = np.array([r["rates"][rate]["reliability"] for r in R[(ds, m)]])  # seeds, bins, 3
            cnt = bins[..., 0].sum(0)
            acc = (bins[..., 0] * bins[..., 1]).sum(0) / np.maximum(cnt, 1)
            conf = (bins[..., 0] * bins[..., 2]).sum(0) / np.maximum(cnt, 1)
            ok = cnt > 0
            c, mk = STYLE[m]
            ece = np.mean([r["rates"][rate]["ece"] for r in R[(ds, m)]])
            ax.plot(conf[ok], acc[ok], color=c, marker=mk, ms=3.6, lw=2.0 if m == "MUSE" else 1.1,
                    label=f"{m} (ECE {100*ece:.1f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{ds}, {int(float(rate)*100)}% missing", fontsize=8.5)
        ax.set_xlabel("confidence")
        if j == 0:
            ax.set_ylabel("accuracy")
        ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_reliability")


def load_mri():
    R = collections.defaultdict(list)
    for f in glob.glob("results/brats/final_*.json"):
        r = json.load(open(f))
        R[r["model"]].append(r)
    return R


def fig_mri_patterns(R):
    if "MUSE" not in R:
        return
    pats = list(R["MUSE"][0]["test"].keys())
    others = [m for m in MAIN if m in R and m != "MUSE"]
    score = lambda m, p: np.mean([np.mean(list(r["test"][p]["dice"].values())) for r in R[m]])
    best = sorted(others, key=lambda m: -np.mean([score(m, p) for p in pats]))[:3]
    models = ["MUSE"] + best
    pats = sorted(pats, key=lambda p: (p.count("+"), p))
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.8 / len(models)
    xs = np.arange(len(pats))
    for i, m in enumerate(models):
        c, _ = STYLE[m]
        ax.bar(xs + (i - (len(models) - 1) / 2) * w, [score(m, p) for p in pats], width=w * 0.92,
               color=c, label=m, zorder=3)
    short = {"FLAIR": "F", "T1": "T1", "T1gd": "T1c", "T2": "T2"}
    ax.set_xticks(xs)
    ax.set_xticklabels(["+".join(short[s] for s in p.split("+")) for p in pats], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("mean Dice (WT, TC, ET)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(models), fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_mri_patterns")


def fig_mri_uncertainty(R):
    if "MUSE" not in R:
        return
    by_k = collections.defaultdict(list)
    for r in R["MUSE"]:
        for p, v in r["test"].items():
            by_k[v["n_modalities"]].append(v["node"]["unc"])
    ks = sorted(by_k)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.boxplot([by_k[k] for k in ks], positions=ks, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#c9dcf3", edgecolor=STYLE["MUSE"][0]),
               medianprops=dict(color=STYLE["MUSE"][0], lw=1.6), whiskerprops=dict(color=MUTED),
               capprops=dict(color=MUTED), flierprops=dict(marker="o", ms=2.5, mec=MUTED))
    ax.set_xlabel("number of available sequences")
    ax.set_ylabel("mean uncertainty u = C/S")
    ax.set_xticks(ks)
    fig.tight_layout()
    save(fig, "fig_mri_uncertainty")


def fig_order():
    f = "results/order_dependence.json"
    if not os.path.exists(f):
        return
    D = json.load(open(f))
    rules = ["MUSE-sum", "dempster", "yager", "pcr5", "average"]
    names = {"MUSE-sum": "MUSE (sum)", "dempster": "Dempster", "yager": "Yager", "pcr5": "PCR5", "average": "averaging"}
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    datasets = [d for d in ("Cora", "CiteSeer", "Photo") if d in D]
    w = 0.8 / len(datasets)
    for i, ds in enumerate(datasets):
        vals = [max(D[ds][r]["prob"], 1e-17) for r in rules]
        ax.bar(np.arange(len(rules)) + (i - (len(datasets) - 1) / 2) * w, vals, width=w * 0.92,
               color=list(STYLE.values())[i][0], label=ds, zorder=3)
    ax.set_yscale("log")
    ax.axhline(1e-15, color=MUTED, lw=0.8, ls=":")
    ax.text(len(rules) - 0.5, 1.6e-15, "float64 precision", fontsize=6.5, color=MUTED, ha="right")
    ax.set_xticks(range(len(rules))); ax.set_xticklabels([names[r] for r in rules], fontsize=7, rotation=20)
    ax.set_ylabel("max |Δ fused belief| across orders")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_order")


if __name__ == "__main__":
    R = load_final()
    fig_missing_rate(R)
    fig_reliability(R)
    M = load_mri()
    fig_mri_patterns(M)
    fig_mri_uncertainty(M)
    fig_order()
    print("figures written to", OUT)

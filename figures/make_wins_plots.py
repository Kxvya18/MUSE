"""Where MUSE wins: MRI plots that put MUSE in colour and every other method in grey.

  wins_forest.png           MUSE minus each baseline on the four metrics MUSE leads, 95% bootstrap CI, Wilcoxon stars
  wins_error_detection.png  per-patient error-detection AUROC of each method's uncertainty
  wins_flair_only.png       mean Dice when only FLAIR is available
  wins_difficulty.png       mean Dice per patient-difficulty tercile (difficulty = mean Dice over all methods)
  wins_panel.png            the four views above in one 2 x 2 figure (paper / slides)

Per-patient scores are averaged over sequence patterns; stars: * p<0.05, ** p<0.01, *** p<0.001
(paired Wilcoxon over the held-out patients).
usage: python figures/make_wins_plots.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

OUT = "figures/highlights"
RUNS = "results/mri3d_runs"
BASE = ["mmFormer", "RFNet", "HeMIS", "TMC", "LateMean", "UNet-MD"]
METHODS = ["MUSE"] + BASE
LABEL = {"mmFormer": "mmFormer-style", "RFNet": "RFNet-style"}
MUSE_C, MUSE_LIGHT, GREY, GREY_D = "#2a78d6", "#a9c8ee", "#b4b7bd", "#6b6f78"
INK, GRID = "#1b1f3b", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GREY_D, "axes.labelcolor": INK,
                     "xtick.color": GREY_D, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK})
lab = lambda m: LABEL.get(m, m)
rng = np.random.default_rng(0)

R = {m: json.load(open(f"{RUNS}/{m}_s0/test.json")) for m in METHODS}
cases = sorted(set.intersection(*[set(v) for v in R.values()]))
pats = list(R["MUSE"][cases[0]])


def per_patient(m, key=None, keep=lambda p: True):
    """One score per patient, averaged over the selected patterns (key=None: mean Dice of WT/TC/ET)."""
    sel = [p for p in pats if keep(p)]
    if key:
        return np.array([np.nanmean([R[m][c][p][key] for p in sel]) for c in cases])
    return np.array([np.mean([np.mean([R[m][c][p][f"dice_{r}"] for r in ("WT", "TC", "ET")]) for p in sel])
                     for c in cases])


mean_dice = {m: per_patient(m) for m in METHODS}
difficulty = np.mean([mean_dice[m] for m in METHODS], 0)
hard = difficulty <= np.quantile(difficulty, 1 / 3)
METRICS = {  # name -> (per-patient scores, patient mask)
    "Error-detection AUROC": ({m: per_patient(m, "auroc_err") for m in METHODS}, slice(None)),
    "Dice, FLAIR only": ({m: per_patient(m, keep=lambda p: p == "FLAIR") for m in METHODS}, slice(None)),
    "Whole-tumour Dice": ({m: per_patient(m, "dice_WT") for m in METHODS}, slice(None)),
    "Dice, hardest third of patients": (mean_dice, hard),
}


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def boot_ci(x, n=4000):
    x = x[np.isfinite(x)]
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def paired(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    return d.mean(), boot_ci(d), wilcoxon(a[ok], b[ok]).pvalue


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def forest(ax_list):
    for ax, (name, (S, mask)) in zip(ax_list, METRICS.items()):
        rows = [(b, *paired(S["MUSE"][mask], S[b][mask])) for b in BASE]
        for i, (b, d, ci, p) in enumerate(rows):
            win = d > 0 and p < 0.05
            c = MUSE_C if win else GREY
            ax.plot(ci, [i, i], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(d, i, "o", ms=7, color=c, mec="white", mew=1.2, zorder=3)
            ax.text(ci[1], i, f"  {stars(p)}", va="center", fontsize=9, color=MUSE_C if win else GREY_D,
                    fontweight="bold" if win else "normal")
        ax.axvline(0, color=GREY_D, lw=0.9, ls="--")
        ax.set_yticks(range(len(BASE)), [f"vs {lab(b)}" for b in BASE])
        ax.invert_yaxis()
        ax.set_title(name, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_xlabel("MUSE − baseline  (95% bootstrap CI)")
        ax.set_xlim(right=ax.get_xlim()[1] + 0.25 * np.ptp(ax.get_xlim()))


def ranked_bars(ax, S, title, ylabel, mask=slice(None), lo=None):
    means = {m: np.nanmean(S[m][mask]) for m in METHODS}
    order = sorted(METHODS, key=lambda m: -means[m])
    for i, m in enumerate(order):
        ci = boot_ci(S[m][mask])
        c = MUSE_C if m == "MUSE" else GREY
        ax.bar(i, means[m], color=c, width=0.68, zorder=2)
        ax.errorbar(i, means[m], yerr=[[means[m] - ci[0]], [ci[1] - means[m]]], color=INK, lw=1, capsize=3, zorder=3)
        txt = f"{means[m]:.3f}"
        if m != "MUSE":
            txt += "\n" + stars(paired(S["MUSE"][mask], S[m][mask])[2])
        ax.text(i, ci[1] + 0.004, txt, ha="center", va="bottom", fontsize=8,
                color=MUSE_C if m == "MUSE" else GREY_D, fontweight="bold" if m == "MUSE" else "normal")
    ax.set_xticks(range(len(order)), [lab(m) for m in order], rotation=30, ha="right")
    for t, m in zip(ax.get_xticklabels(), order):
        if m == "MUSE":
            t.set_color(MUSE_C); t.set_fontweight("bold")
    top = max(boot_ci(S[m][mask])[1] for m in METHODS)
    ax.set_ylim(lo if lo is not None else 0, top + 0.12 * (top - (lo or 0)))
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def error_detection(ax):
    S = METRICS["Error-detection AUROC"][0]
    order = sorted(METHODS, key=lambda m: -np.nanmean(S[m]))
    for i, m in enumerate(order):
        x = S[m][np.isfinite(S[m])]
        c = MUSE_C if m == "MUSE" else GREY
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(x)), x, s=9, color=MUSE_LIGHT if m == "MUSE" else "#dcdde0",
                   edgecolor="none", zorder=1)
        ax.boxplot(x, positions=[i], widths=0.5, showfliers=False, patch_artist=True, zorder=2,
                   boxprops=dict(facecolor="none", edgecolor=c, lw=1.6), medianprops=dict(color=c, lw=2.2),
                   whiskerprops=dict(color=c, lw=1.2), capprops=dict(color=c, lw=1.2))
        lbl = f"{np.mean(x):.3f}" + ("" if m == "MUSE" else "\n" + stars(paired(S["MUSE"], S[m])[2]))
        ax.text(i, 1.005, lbl, ha="center", va="bottom", fontsize=8, color=MUSE_C if m == "MUSE" else GREY_D,
                fontweight="bold" if m == "MUSE" else "normal")
    ax.set_xticks(range(len(order)), [lab(m) for m in order], rotation=30, ha="right")
    for t, m in zip(ax.get_xticklabels(), order):
        if m == "MUSE":
            t.set_color(MUSE_C); t.set_fontweight("bold")
    lo = min(np.nanpercentile(S[m], 5) for m in METHODS)
    ax.set_ylim(lo - 0.02, 1.07)
    ax.set_yticks([t for t in ax.get_yticks() if t <= 1.0001])
    ax.set_ylabel("AUROC: uncertainty flags wrong voxels")
    ax.set_title("Knowing when it is wrong", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def difficulty_curve(ax):
    edges = np.quantile(difficulty, [0, 1 / 3, 2 / 3, 1])
    bins = [(difficulty >= edges[i]) & (difficulty <= edges[i + 1]) for i in range(3)]
    xs = np.arange(3)
    for m in METHODS[::-1]:
        y = [mean_dice[m][b].mean() for b in bins]
        is_m = m == "MUSE"
        ax.plot(xs, y, "-o", color=MUSE_C if is_m else GREY, lw=2.8 if is_m else 1.1, ms=7 if is_m else 4,
                zorder=3 if is_m else 2, label="MUSE" if is_m else ("baselines" if m == BASE[0] else None))
    ax.legend(frameon=False, loc="lower right")
    # hardest-third gap to the best baseline
    yb = max(mean_dice[b][bins[0]].mean() for b in BASE)
    ym = mean_dice["MUSE"][bins[0]].mean()
    ax.annotate(f"+{ym - yb:.3f} over the\nbest baseline", xy=(0, ym), xytext=(0.18, ym - 0.12), fontsize=8.5,
                color=MUSE_C, arrowprops=dict(arrowstyle="-", color=MUSE_C, lw=1))
    ax.set_xticks(xs, [f"hardest third\n(n={bins[0].sum()})", f"middle\n(n={bins[1].sum()})",
                       f"easiest third\n(n={bins[2].sum()})"])
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel("mean Dice (WT/TC/ET, 15 patterns)")
    ax.set_title("Largest margin on the hardest patients", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    forest(axes.ravel())
    fig.suptitle("MUSE vs every baseline on the metrics where it leads (blue = MUSE significantly better)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "wins_forest")

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    error_detection(ax)
    save(fig, "wins_error_detection")

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ranked_bars(ax, METRICS["Dice, FLAIR only"][0], "Only FLAIR available", "mean Dice (WT/TC/ET)")
    save(fig, "wins_flair_only")

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    difficulty_curve(ax)
    save(fig, "wins_difficulty")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    error_detection(axes[0, 0])
    ranked_bars(axes[0, 1], METRICS["Dice, FLAIR only"][0], "Only FLAIR available", "mean Dice (WT/TC/ET)")
    ranked_bars(axes[1, 0], METRICS["Whole-tumour Dice"][0], "Whole-tumour Dice (15 patterns)", "Dice WT", lo=0.6)
    ranked_bars(axes[1, 1], mean_dice, f"Hardest third of patients (n={hard.sum()})", "mean Dice (WT/TC/ET)",
                mask=hard, lo=0.25)
    for ax, t in zip(axes.ravel(), "abcd"):
        ax.text(-0.12, 1.06, f"({t})", transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(h_pad=2.5)
    save(fig, "wins_panel")
    print(f"{len(cases)} patients; written to {OUT}/wins_*")


if __name__ == "__main__":
    main()

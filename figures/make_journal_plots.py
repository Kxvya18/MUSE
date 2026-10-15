"""Journal-style MRI result figures (MUSE in colour, baselines muted).

  journal_bars.png     grouped bars per metric, dots = 5 disjoint patient folds, 95% bootstrap CI,
                       bracket = paired Wilcoxon p of MUSE vs the strongest baseline on that metric
  journal_gain_map.png MUSE minus each baseline, mean Dice for all 15 sequence patterns (dot = p < 0.05)
  journal_stack.png    stacked panels over patient difficulty: (a) Dice, (b) MUSE gain, (c) MUSE uncertainty,
                       (d) MUSE gain over the mean baseline by number of available sequences (filled contours)

usage: python figures/make_journal_plots.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.stats import wilcoxon

OUT = "figures/highlights"
RUNS = "results/mri3d_runs"
BASE = ["mmFormer", "RFNet", "HeMIS", "TMC", "LateMean", "UNet-MD"]
METHODS = ["MUSE"] + BASE
LABEL = {"mmFormer": "mmFormer-style", "RFNet": "RFNet-style"}
FILL = {"MUSE": "#2a78d6", "mmFormer": "#f2b8a2", "RFNet": "#a8dcc4", "HeMIS": "#f6d98e", "TMC": "#f1c4d8",
        "LateMean": "#c9e0b4", "UNet-MD": "#ffffff"}
INK, MUTED = "#1b1f3b", "#55575e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.9, "xtick.direction": "out", "ytick.direction": "out"})
lab = lambda m: LABEL.get(m, m)
rng = np.random.default_rng(0)

R = {m: json.load(open(f"{RUNS}/{m}_s0/test.json")) for m in METHODS}
cases = sorted(set.intersection(*[set(v) for v in R.values()]))
pats = sorted(R["MUSE"][cases[0]], key=lambda p: (p.count("+"), p))
nseq = np.array([p.count("+") + 1 for p in pats])


def grid(key):
    """[method] -> array [patients, patterns]; key=None is mean Dice of WT/TC/ET."""
    get = (lambda d: np.mean([d[f"dice_{r}"] for r in ("WT", "TC", "ET")])) if key is None else (lambda d: d[key])
    return {m: np.array([[get(R[m][c][p]) for p in pats] for c in cases], dtype=float) for m in METHODS}


DICE = grid(None)
difficulty = np.mean([DICE[m].mean(1) for m in METHODS], 0)
hard = difficulty <= np.quantile(difficulty, 1 / 3)
flair = [pats.index("FLAIR")]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", dpi=260, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def boot_ci(x, n=4000):
    return np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5])


def fmt_p(p):
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}" if p < 0.01 else f"p = {p:.3f}"


def bars():
    au = grid("auroc_err")
    groups = [  # title, per-patient scores per method (x100)
        ("Whole tumour\nDice", {m: 100 * grid("dice_WT")[m].mean(1) for m in METHODS}),
        ("FLAIR only\nDice", {m: 100 * DICE[m][:, flair].mean(1) for m in METHODS}),
        ("Hardest third\nDice", {m: 100 * DICE[m][hard].mean(1) for m in METHODS}),
        ("Error detection\nAUROC", {m: 100 * np.nanmean(au[m], 1) for m in METHODS}),
    ]
    folds = np.array_split(rng.permutation(len(cases)), 5)
    w, gap = 0.12, 0.35
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    centers = []
    for g, (title, S) in enumerate(groups):
        x0 = g * (len(METHODS) * w + gap)
        xs = x0 + np.arange(len(METHODS)) * w
        centers.append(xs.mean())
        tops = []
        for x, m in zip(xs, METHODS):
            v = S[m]
            fold_idx = [f[f < len(v)] for f in folds] if len(v) == len(cases) else np.array_split(rng.permutation(len(v)), 5)
            mean, ci = v.mean(), boot_ci(v)
            is_m = m == "MUSE"
            ax.bar(x, mean, w * 0.9, color=FILL[m], edgecolor=INK if not is_m else "#174f94", lw=0.8, zorder=2)
            ax.errorbar(x, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], color=INK, lw=0.9, capsize=2.5, zorder=3)
            fv = [v[f].mean() for f in fold_idx]
            ax.scatter(x + rng.uniform(-w * 0.18, w * 0.18, 5), fv, s=13, facecolor="white", edgecolor=INK, lw=0.7, zorder=4)
            tops.append(max(ci[1], max(fv)))
        # bracket: MUSE vs strongest baseline in this group; stars line: number of baselines MUSE beats significantly
        best = max(BASE, key=lambda b: S[b].mean())
        p_best = wilcoxon(S["MUSE"], S[best]).pvalue
        n_sig = sum(S["MUSE"].mean() > S[b].mean() and wilcoxon(S["MUSE"], S[b]).pvalue < 0.05 for b in BASE)
        bi = METHODS.index(best)
        y = max(tops) + 3
        ax.plot([xs[0], xs[0], xs[bi], xs[bi]], [y - 1, y, y, y - 1], color=INK, lw=0.8)
        win = S["MUSE"].mean() > S[best].mean()
        ax.text((xs[0] + xs[bi]) / 2, y + 0.6, fmt_p(p_best) if win else "n.s. (tie)", ha="center", fontsize=8.5,
                color=INK)
        ax.text(centers[-1], y + 4.2, f"MUSE > {n_sig}/6 baselines", ha="center", fontsize=8.5,
                color=FILL["MUSE"] if n_sig >= 5 else MUTED, fontweight="bold" if n_sig >= 5 else "normal")
    ax.set_xticks(centers, [t for t, _ in groups])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("score (× 100)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=FILL[m], edgecolor=INK, lw=0.8) for m in METHODS]
    ax.legend(handles, [lab(m) for m in METHODS], ncol=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), handlelength=1.1, columnspacing=1.4)
    for t in ax.get_legend().get_texts()[:1]:
        t.set_fontweight("bold"); t.set_color(FILL["MUSE"])
    save(fig, "journal_bars")


def gain_map():
    G = np.array([(DICE["MUSE"] - DICE[b]).mean(0) for b in BASE])
    sig = np.array([[wilcoxon(DICE["MUSE"][:, j], DICE[b][:, j]).pvalue < 0.05 and G[i, j] > 0
                     for j in range(len(pats))] for i, b in enumerate(BASE)])
    lim = np.abs(G).max()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(G, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    yy, xx = np.nonzero(sig)
    ax.scatter(xx, yy, s=10, color="white", edgecolor=INK, lw=0.5, zorder=3)
    ax.set_xticks(range(len(pats)), [p.replace("+", "\n") for p in pats], fontsize=7.2)
    ax.set_yticks(range(len(BASE)), [f"vs {lab(b)}" for b in BASE])
    ax.set_xticks(np.arange(-0.5, len(pats)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BASE)), minor=True)
    ax.grid(which="minor", color="#9a9a9a", lw=0.6)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(True)
    for k in (4, 10, 14):  # separators between 1-, 2-, 3-, 4-sequence blocks
        ax.axvline(k - 0.5, color=INK, lw=1.4)
    for k, (a, b) in zip(range(1, 5), [(0, 3), (4, 9), (10, 13), (14, 14)]):
        ax.text((a + b) / 2, -0.62, f"{k} sequence{'s' if k > 1 else ''}", ha="center", fontsize=9, color=INK)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("MUSE − baseline, mean Dice")
    ax.set_xlabel("\navailable sequences   (dot: MUSE significantly better, paired Wilcoxon p < 0.05)")
    ax.set_title("Where MUSE gains over each baseline", loc="left", fontweight="bold", color=INK, pad=26)
    save(fig, "journal_gain_map")


def stack():
    order = np.argsort(difficulty)[::-1]  # easiest -> hardest
    pct = np.linspace(0, 100, len(order))
    sm = lambda v: uniform_filter1d(v, 13, mode="nearest")
    m_d = DICE["MUSE"].mean(1)[order]
    base = np.array([DICE[b].mean(1)[order] for b in BASE])
    au = grid("auroc_err")
    m_au = np.nanmean(au["MUSE"], 1)[order]
    b_au = np.array([np.nanmean(au[bb], 1)[order] for bb in BASE])
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 0.8, 0.8, 1.5], "hspace": 0.08})
    a, b, c, d = axes
    a.fill_between(pct, sm(base.min(0)), sm(base.max(0)), color="#d9d9d9", lw=0, label="baseline range")
    a.plot(pct, sm(m_d), color=FILL["MUSE"], lw=1.8, label="MUSE")
    a.set_ylabel("Dice")
    a.legend(frameon=False, loc="lower left", fontsize=8.5)
    gain = m_d - base.mean(0)
    b.plot(pct, sm(gain), color=INK, lw=1.1)
    b.fill_between(pct, 0, sm(gain), where=sm(gain) > 0, color=FILL["MUSE"], alpha=0.35, lw=0)
    b.axhline(0, color=MUTED, ls="--", lw=0.8)
    hard_gain = gain[pct >= 200 / 3].mean()
    b.axvspan(200 / 3, 100, color=FILL["MUSE"], alpha=0.07, lw=0)
    b.text(98, 0.004, f"hardest third: +{hard_gain:.3f}", ha="right", va="bottom", fontsize=8.5,
           color=FILL["MUSE"], fontweight="bold")
    b.set_ylabel("MUSE − mean\nbaseline")
    c.fill_between(pct, sm(b_au.min(0)), sm(b_au.max(0)), color="#d9d9d9", lw=0)
    c.plot(pct, sm(m_au), color=FILL["MUSE"], lw=1.8)
    c.yaxis.set_label_position("right"); c.yaxis.tick_right()
    c.spines["right"].set_visible(True); c.spines["left"].set_visible(False)
    c.set_ylabel("error-detection\nAUROC")
    # (d) gain over the mean baseline, by patient difficulty x number of available sequences
    gain_pk = np.array([[(DICE["MUSE"][:, nseq == k] - np.mean([DICE[bb][:, nseq == k] for bb in BASE], 0)).mean(1)[order][i]
                         for i in range(len(order))] for k in range(1, 5)])
    Z = gaussian_filter(gain_pk, (0.6, 4))
    lim = np.ceil(np.abs(Z).max() * 100) / 100
    cf = d.contourf(pct, np.arange(1, 5), Z, levels=np.linspace(-lim, lim, 15), cmap="RdBu")
    cs = d.contour(pct, np.arange(1, 5), Z, levels=[0.02, 0.05], colors=INK, linewidths=0.8)
    d.clabel(cs, fmt={0.02: "+0.02", 0.05: "+0.05"}, fontsize=7.5)
    d.set_yticks(range(1, 5))
    d.set_ylabel("available sequences")
    d.set_xlabel("patients ordered from easiest to hardest (percentile)")
    cax = fig.add_axes([0.93, 0.11, 0.02, 0.25])
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_ticks(np.round(np.linspace(-lim, lim, 5), 3))
    cbar.set_label("MUSE − mean baseline\n(Dice)")
    for ax, t in zip(axes, "abcd"):
        ax.text(0.01, 0.95, f"({t})", transform=ax.transAxes, va="top", fontsize=10, color=INK)
    a.set_xlim(0, 100)
    save(fig, "journal_stack")


if __name__ == "__main__":
    bars()
    gain_map()
    stack()
    print(f"{len(cases)} patients -> {OUT}/journal_*")

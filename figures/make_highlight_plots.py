"""Highlight plots of the final MUSE results (MRI: results/mri3d_runs; graphs: results/final).

  radar_mri.png              per-axis normalised profile: MUSE vs the strongest baselines
  heatmap_patterns.png       mean Dice for every method x all 15 sequence combinations
  robustness_curve.png       Dice vs number of available sequences (95% CI over patients)
  paired_scatter.png         per-patient MUSE vs mmFormer-style Dice
  error_detection.png        per-patient error-detection AUROC of each method's uncertainty
  contributions_donut.png    MUSE evidence share per sequence on tumour voxels
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figures/highlights"
RUNS = "results/mri3d_runs"
METHODS = ["MUSE", "mmFormer", "RFNet", "HeMIS", "TMC", "LateMean", "UNet-MD"]
LABEL = {"mmFormer": "mmFormer-style", "RFNet": "RFNet-style"}
COL = {"MUSE": "#2a78d6", "mmFormer": "#eb6834", "RFNet": "#1baf7a", "HeMIS": "#eda100",
       "TMC": "#e87ba4", "LateMean": "#008300", "UNet-MD": "#4a3aa7"}
SEQ_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
INK, MUTED, GRID = "#1b1f3b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
lab = lambda m: LABEL.get(m, m)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


R = {m: json.load(open(f"{RUNS}/{m}_s0/test.json")) for m in METHODS}
cases = sorted(set.intersection(*[set(v) for v in R.values()]))
pats = sorted(R["MUSE"][cases[0]], key=lambda p: (-p.count("+"), p))
dice = lambda m, c, p: np.mean([R[m][c][p][f"dice_{r}"] for r in ("WT", "TC", "ET")])
per_patient = {m: np.array([[dice(m, c, p) for p in pats] for c in cases]) for m in METHODS}  # cases x patterns


def radar():
    axes_def = [("WT Dice", lambda m: np.mean([R[m][c][p]["dice_WT"] for c in cases for p in pats]), True),
                ("TC Dice", lambda m: np.mean([R[m][c][p]["dice_TC"] for c in cases for p in pats]), True),
                ("ET Dice", lambda m: np.mean([R[m][c][p]["dice_ET"] for c in cases for p in pats]), True),
                ("Error detection\n(AUROC)", lambda m: np.nanmean([R[m][c][p]["auroc_err"] for c in cases for p in pats]), True),
                ("Calibration\n(low ECE)", lambda m: np.mean([R[m][c][p]["ece"] for c in cases for p in pats]), False),
                ("Single-sequence\nrobustness", lambda m: per_patient[m][:, [i for i, p in enumerate(pats) if "+" not in p]].mean(), True)]
    show = ["MUSE", "mmFormer", "RFNet", "HeMIS"]
    vals = np.array([[f(m) for _, f, _ in axes_def] for m in show])
    lo, hi = vals.min(0), vals.max(0)
    norm = (vals - lo) / np.where(hi > lo, hi - lo, 1)
    for j, (_, _, up) in enumerate(axes_def):
        if not up:
            norm[:, j] = 1 - norm[:, j]
    norm = 0.2 + 0.8 * norm  # keep the worst method visible
    ang = np.linspace(0, 2 * np.pi, len(axes_def), endpoint=False).tolist()
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = fig.add_subplot(polar=True)
    for i, m in enumerate(show):
        v = norm[i].tolist()
        ax.plot(ang + ang[:1], v + v[:1], color=COL[m], lw=2.6 if m == "MUSE" else 1.5, label=lab(m), zorder=5 if m == "MUSE" else 2)
        ax.fill(ang + ang[:1], v + v[:1], color=COL[m], alpha=0.18 if m == "MUSE" else 0.04)
    ax.set_xticks(ang); ax.set_xticklabels([a for a, _, _ in axes_def], fontsize=10, color=INK)
    ax.set_yticks([0.2, 0.6, 1.0]); ax.set_yticklabels(["worst", "", "best"], fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.05); ax.spines["polar"].set_color(GRID); ax.grid(color=GRID)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    ax.set_title("Where each method is strongest\n(each axis scaled from worst to best of these four)", pad=26, color=INK, fontsize=12)
    save(fig, "radar_mri")


def heatmap():
    M = np.array([[per_patient[m][:, j].mean() for j in range(len(pats))] for m in METHODS])
    short = {"FLAIR": "F", "T1": "T1", "T1gd": "T1c", "T2": "T2"}
    fig, ax = plt.subplots(figsize=(11, 3.9))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0.1, vmax=0.8, aspect="auto")
    best = M.argmax(0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] > 0.55 else INK, fontweight="bold" if best[j] == i else "normal")
            if best[j] == i:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#e5534b", lw=2))
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([lab(m) for m in METHODS])
    ax.set_xticks(range(len(pats))); ax.set_xticklabels(["+".join(short[s] for s in p.split("+")) for p in pats], rotation=45, ha="right")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("mean Dice")
    ax.set_title(f"Mean Dice for every available-sequence combination ({len(cases)} test patients; red box = best)", color=INK)
    save(fig, "heatmap_patterns")


def robustness():
    sizes = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in METHODS:
        means, cis = [], []
        for k in sizes:
            cols = [j for j, p in enumerate(pats) if p.count("+") + 1 == k]
            v = per_patient[m][:, cols].mean(1)
            means.append(v.mean()); cis.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        means, cis = np.array(means), np.array(cis)
        ax.plot(sizes, means, marker="o", ms=6 if m == "MUSE" else 4, lw=2.8 if m == "MUSE" else 1.4, color=COL[m], label=lab(m), zorder=5 if m == "MUSE" else 2)
        if m == "MUSE":
            ax.fill_between(sizes, means - cis, means + cis, color=COL[m], alpha=0.15, lw=0)
    ax.set_xticks(sizes); ax.set_xlabel("number of available MRI sequences"); ax.set_ylabel("mean Dice (WT, TC, ET)")
    ax.grid(axis="y", color=GRID); ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower right")
    ax.set_title("Accuracy as sequences go missing (shaded: MUSE 95% CI)", color=INK, fontsize=11)
    save(fig, "robustness_curve")


def paired():
    a, b = per_patient["MUSE"].mean(1), per_patient["mmFormer"].mean(1)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    win = a > b
    ax.scatter(b[win], a[win], s=26, color=COL["MUSE"], alpha=0.8, label=f"MUSE better ({win.sum()} patients)", edgecolor="white", lw=0.5)
    ax.scatter(b[~win], a[~win], s=26, color=COL["mmFormer"], alpha=0.8, label=f"mmFormer-style better ({(~win).sum()})", edgecolor="white", lw=0.5)
    lim = [min(a.min(), b.min()) - 0.03, max(a.max(), b.max()) + 0.03]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("mmFormer-style  (per-patient mean Dice)"); ax.set_ylabel("MUSE  (per-patient mean Dice)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left"); ax.grid(color=GRID)
    ax.set_title("Patient by patient: MUSE vs mmFormer-style", color=INK, fontsize=11)
    save(fig, "paired_scatter")


def error_detection():
    data = [np.array([np.nanmean([R[m][c][p]["auroc_err"] for p in pats]) for c in cases]) for m in METHODS]
    order = np.argsort([-np.nanmean(d) for d in data])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bp = ax.boxplot([data[i] for i in order], vert=False, widths=0.55, patch_artist=True, showfliers=False)
    for patch, i in zip(bp["boxes"], order):
        patch.set_facecolor(COL[METHODS[i]]); patch.set_alpha(0.75); patch.set_edgecolor(COL[METHODS[i]])
    for med in bp["medians"]:
        med.set_color("white"); med.set_linewidth(1.8)
    ax.set_yticklabels([lab(METHODS[i]) for i in order]); ax.invert_yaxis()
    ax.set_xlabel("error-detection AUROC of the model's uncertainty (per patient; 1 = perfect)")
    ax.grid(axis="x", color=GRID)
    ax.set_title("Does the uncertainty point at the model's own mistakes?", color=INK, fontsize=11)
    save(fig, "error_detection")


def donut():
    full = "FLAIR+T1+T1gd+T2"
    c = np.array([R["MUSE"][cc][full]["contrib"] for cc in cases]).mean(0)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    w, _ = ax.pie(c, colors=SEQ_COL, startangle=90, counterclock=False, wedgeprops=dict(width=0.38, edgecolor="white", lw=2))
    for wedge, name, v in zip(w, ["FLAIR", "T1", "T1gd", "T2"], c):
        ang = np.deg2rad((wedge.theta2 + wedge.theta1) / 2)
        ax.text(1.18 * np.cos(ang), 1.18 * np.sin(ang), f"{name}\n{100 * v:.0f}%", ha="center", va="center", fontsize=11, color=INK)
    ax.text(0, 0, "MUSE\nevidence\non tumour", ha="center", va="center", fontsize=11, color=MUTED)
    ax.set_title("Which sequence MUSE relied on", color=INK, fontsize=11)
    save(fig, "contributions_donut")


if __name__ == "__main__":
    radar(); heatmap(); robustness(); paired(); error_detection(); donut()
    print("highlight plots written to", OUT)

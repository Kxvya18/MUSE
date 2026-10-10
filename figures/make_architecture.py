"""MUSE architecture diagram (vector PDF + PNG) for the paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"
C_ENC, C_EV, C_GATE, C_FUSE, C_OUT, C_MISS = "#dbe9f6", "#e3f1e4", "#fdf0d5", "#ece4f7", "#f6e1e1", "#f2f2f2"

fig, ax = plt.subplots(figsize=(8.2, 3.9))
ax.set_xlim(0, 122); ax.set_ylim(0, 53); ax.axis("off")


def box(x, y, w, h, text, fc, ec=INK, ls="-", fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.9, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(x1, y1, x2, y2, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 lw=0.8, color=color, ls=ls, shrinkA=1, shrinkB=1))


mods = ["FLAIR", "T1", "T1gd", "T2"]
ys = [40, 30, 20, 10]
missing = {2}  # illustrate one absent modality
FX, FY = 62, 25.5  # combiner input point
for i, (m, y) in enumerate(zip(mods, ys)):
    absent = i in missing
    box(1, y, 10, 6, m + ("\n(absent)" if absent else ""), C_MISS if absent else "#ffffff",
        ec=MUTED if absent else INK, ls="--" if absent else "-", fs=6.8)
    if absent:
        box(15, y, 40, 6, "no encoder pass\nvacuous mass " + r"$\alpha^{(k)}=\mathbf{1},\ \gamma^{(k)}=0$",
            C_MISS, ec=MUTED, ls="--", fs=7)
        arrow(11.4, y + 3, 14.6, y + 3, MUTED, "--")
        arrow(55.4, y + 3, FX - 0.4, FY, MUTED, "--")
        continue
    box(15, y, 12, 6, "GATv2\nencoder", C_ENC)
    box(30, y, 12, 6, "evidential\nhead", C_EV)
    box(45, y, 10, 6, "gate " + r"$\gamma^{(k)}$", C_GATE)
    arrow(11.4, y + 3, 14.6, y + 3)
    arrow(27.4, y + 3, 29.6, y + 3)
    arrow(42.4, y + 3, 44.6, y + 3)
    arrow(55.4, y + 3, FX - 0.4, FY)

ax.text(28, 49.5, "per-modality evidence " + r"$\alpha^{(k)}=\mathrm{softplus}(\cdot)+1$" + ",  discount " + r"$\gamma^{(k)}\in(0,1)^C$",
        ha="center", fontsize=7.2, color=MUTED)
box(FX, 18, 21, 15, "N-modal combiner\n\n" + r"$\Sigma_{k\in O}\,(\alpha^{(k)}-1)\,\gamma^{(k)}+1$", C_FUSE)
box(FX, 2, 21, 9, "context " + r"$c_i$" + "\n(neighbourhood mean)", C_GATE, fs=7)
ax.annotate("", xy=(55.4, 12.5), xytext=(FX - 0.4, 6.5),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, ls="--", mutation_scale=8))
ax.text(52, 6.5, "to gates", fontsize=6.8, color=MUTED, ha="center")
box(87, 18, 14, 15, "evidence\npropagation\n(PPR)", C_FUSE)
arrow(83.4, FY, 86.6, FY)
outs = [("class label\n" + r"$\mathrm{argmax}_c\ b_c$", 38), ("uncertainty\n" + r"$u=C/S$", 25.5), ("contribution\nweights " + r"$\gamma^{(k)}$", 13)]
for t, y in outs:
    box(105, y - 4.5, 16, 9, t, C_OUT, fs=7)
    arrow(101.4, FY, 104.6, y)
ax.text(113, 47.5, "one forward pass", ha="center", fontsize=7, color=MUTED)
fig.savefig("figures/muse_architecture.pdf", bbox_inches="tight")
fig.savefig("figures/muse_architecture.png", dpi=220, bbox_inches="tight")

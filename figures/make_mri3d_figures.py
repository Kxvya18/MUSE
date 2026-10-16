"""MRI figures for the papers, from real held-out patients and the trained 3D models.

  figures/mri3d/<case>_missing_panel.{pdf,png}  axial slice for 4 sequence patterns: inputs (absent
                                                greyed), ground truth, MUSE, HeMIS, MUSE uncertainty,
                                                dominant sequence per tumour voxel
  figures/mri3d/<case>_triplanar.{pdf,png}      axial / coronal / sagittal: FLAIR, GT, MUSE, uncertainty
  figures/mri3d/<case>_3d.{pdf,png}             3D tumour surfaces: GT vs MUSE (all four / FLAIR only),
                                                MUSE surface coloured by its uncertainty
  figures/mri3d/dice_per_pattern.{pdf,png}      mean Dice per sequence pattern (all models)
  figures/mri3d/uncertainty_vs_sequences.{pdf,png}

usage: python figures/make_mri3d_figures.py [--cases BRATS_a,BRATS_b] [--n 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_mri3d as R  # noqa: E402
from muse3d.data import SEQUENCES, load_case, load_full_labels  # noqa: E402
from muse3d.models import MODELS  # noqa: E402

DATA, RUNS, OUT = "data/mri3d_test1mm", "runs_final", "figures/mri3d"
SEG = ListedColormap([(0, 0, 0, 0), "#2ca02c", "#d62728", "#ffd21f"])  # bg, edema, core, enhancing
SEQ_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # FLAIR, T1, T1gd, T2 (validated palette)
PATTERNS = {"all four": (0, 1, 2, 3), "T1gd absent": (0, 1, 3), "FLAIR + T2": (0, 3), "FLAIR only": (0,)}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_model(name):
    m = MODELS[name]().to(R.DEV)
    m.load_state_dict(torch.load(f"{RUNS}/{name}_s0/best.pt", map_location=R.DEV))
    return m.eval()


def predict(model, x, patterns):
    """{pattern: (pred labels, uncertainty, per-sequence contribution [4,...])} at full resolution."""
    out = R.predict_case(model, x.astype(np.float32), list(patterns))
    return {p: (prob.argmax(0), unc, contrib) for p, (prob, unc, contrib) in out.items()}


def show(ax, img, overlay=None, cmap=None, vmax=None, title=None):
    ax.imshow(np.rot90(img), cmap="gray", vmin=-2, vmax=4)
    if overlay is not None:
        ax.imshow(np.rot90(overlay), cmap=cmap, vmin=0 if vmax else -0.5, vmax=vmax or 3.5,
                  interpolation="nearest", alpha=0.85 if vmax else 0.75)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def missing_panel(case, x, y, brain, muse, hemis):
    sl = int(np.argmax((y > 0).sum((0, 1))))
    pm, ph = predict(muse, x, PATTERNS.values()), predict(hemis, x, PATTERNS.values())
    umax = max(float(pm[p][1][..., sl].max()) for p in PATTERNS.values())
    fig, axes = plt.subplots(len(PATTERNS), 9, figsize=(14, 1.7 * len(PATTERNS) + 0.6))
    for r, (name, p) in enumerate(PATTERNS.items()):
        for k in range(4):
            ax = axes[r, k]
            show(ax, x[k][..., sl], title=SEQUENCES[k] if r == 0 else None)
            if k not in p:
                ax.imshow(np.rot90(np.ones_like(x[k][..., sl])), cmap="gray", vmin=0, vmax=1, alpha=0.82)
                ax.text(0.5, 0.5, "absent", transform=ax.transAxes, ha="center", va="center", color="#444")
        lab = np.where(y > 0, y, 0)[..., sl]
        show(axes[r, 4], x[0][..., sl], lab, SEG, title="ground truth" if r == 0 else None)
        show(axes[r, 5], x[0][..., sl], np.where(pm[p][0] > 0, pm[p][0], 0)[..., sl], SEG, title="MUSE" if r == 0 else None)
        show(axes[r, 6], x[0][..., sl], np.where(ph[p][0] > 0, ph[p][0], 0)[..., sl], SEG, title="HeMIS" if r == 0 else None)
        u = np.where(brain, pm[p][1], np.nan)[..., sl]
        show(axes[r, 7], x[0][..., sl], u, "magma", vmax=umax, title="MUSE uncertainty" if r == 0 else None)
        # dominant sequence: which sequence contributed most evidence at each predicted tumour voxel
        c = pm[p][2][..., sl]
        dom = np.where(pm[p][0][..., sl] > 0, c.argmax(0).astype(float), np.nan)
        axes[r, 8].imshow(np.rot90(x[0][..., sl]), cmap="gray", vmin=-2, vmax=4)
        axes[r, 8].imshow(np.rot90(dom), cmap=ListedColormap(SEQ_COL), vmin=-0.5, vmax=3.5, interpolation="nearest", alpha=0.85)
        axes[r, 8].set_xticks([]); axes[r, 8].set_yticks([])
        if r == 0:
            axes[r, 8].set_title("dominant sequence", fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=8.5)
    fig.legend(handles=[Patch(color="#2ca02c", label="edema"), Patch(color="#d62728", label="tumour core"),
                        Patch(color="#ffd21f", label="enhancing tumour")] +
               [Patch(color=SEQ_COL[k], label=f"{SEQUENCES[k]} dominant") for k in range(4)],
               loc="lower center", ncol=7, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, f"{case}_missing_panel")
    return pm


def triplanar(case, x, y, brain, pm):
    pred, unc, _ = pm[(0, 1, 2, 3)]
    c = np.argwhere(y > 0).mean(0).astype(int)
    views = {"axial": lambda a: a[:, :, c[2]], "coronal": lambda a: a[:, c[1], :], "sagittal": lambda a: a[c[0], :, :]}
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6))
    umax = float(unc.max())
    for r, (vname, cut) in enumerate(views.items()):
        show(axes[r, 0], cut(x[0]), title="FLAIR" if r == 0 else None)
        show(axes[r, 1], cut(x[0]), cut(np.where(y > 0, y, 0)), SEG, title="ground truth" if r == 0 else None)
        show(axes[r, 2], cut(x[0]), cut(np.where(pred > 0, pred, 0)), SEG, title="MUSE" if r == 0 else None)
        show(axes[r, 3], cut(x[0]), cut(np.where(brain, unc, np.nan)), "magma", vmax=umax,
             title="MUSE uncertainty" if r == 0 else None)
        axes[r, 0].set_ylabel(vname, fontsize=8.5)
    fig.tight_layout()
    save(fig, f"{case}_triplanar")


def mesh(ax, mask, colour=None, values=None, vmax=None, alpha=0.9):
    """Shaded marching-cubes surface of a binary mask; coloured by `values` if given."""
    if mask.sum() < 50:
        return None
    verts, faces, normals, _ = marching_cubes(mask.astype(np.float32), 0.5, step_size=2)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    shade = 0.72 + 0.28 * np.clip(n @ np.array([0.4, -0.5, 0.75]) / 1.05, -1, 1)  # simple Lambert light
    if values is not None:
        idx = np.clip(np.round(verts).astype(int), 0, np.array(mask.shape) - 1)
        v = values[tuple(idx.T)][faces].mean(1) / vmax
        rgb = plt.cm.magma(np.clip(v, 0, 1))[:, :3]
    else:
        rgb = np.tile(matplotlib.colors.to_rgb(colour), (len(faces), 1))
    face = np.c_[rgb * shade[:, None], np.full(len(faces), alpha)]
    pc = Poly3DCollection(tri, linewidths=0)
    pc.set_facecolor(face)
    ax.add_collection3d(pc)
    return verts


def frame(ax, verts, title):
    lo, hi = verts.min(0), verts.max(0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 0.62
    ax.set_xlim(mid[0] - half, mid[0] + half); ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)


def regions(ax, lab, title):
    """Nested sub-region surfaces: whole tumour (translucent), core, enhancing."""
    v = mesh(ax, lab > 0, "#2ca02c", alpha=0.18)
    mesh(ax, (lab == 2) | (lab == 3), "#d62728", alpha=0.55)
    mesh(ax, lab == 3, "#ffd21f", alpha=0.95)
    frame(ax, v, title)


def render3d(case, y, pm):
    full, flair = pm[(0, 1, 2, 3)], pm[(0,)]
    on_surface = np.concatenate([full[1][full[0] > 0], flair[1][flair[0] > 0]])
    vmax = float(np.percentile(on_surface, 98))  # colour range of the uncertainty actually on the tumours
    fig = plt.figure(figsize=(13, 3.8))
    regions(fig.add_subplot(1, 4, 1, projection="3d"), y, "ground truth")
    regions(fig.add_subplot(1, 4, 2, projection="3d"), full[0], "MUSE, all four sequences")
    ax = fig.add_subplot(1, 4, 3, projection="3d")
    frame(ax, mesh(ax, full[0] > 0, values=full[1], vmax=vmax), "MUSE uncertainty on tumour surface\n(all four sequences)")
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    frame(ax, mesh(ax, flair[0] > 0, values=flair[1], vmax=vmax), "MUSE uncertainty on tumour surface\n(FLAIR only)")
    sm = plt.cm.ScalarMappable(cmap="magma"); sm.set_array([0, vmax])
    fig.colorbar(sm, ax=fig.axes, fraction=0.012, pad=0.0, label="uncertainty u")
    fig.subplots_adjust(left=0, right=0.94, wspace=0)
    fig.legend(handles=[Patch(color="#2ca02c", alpha=0.4, label="edema (whole tumour)"),
                        Patch(color="#d62728", label="tumour core"), Patch(color="#ffd21f", label="enhancing tumour")],
               loc="lower left", ncol=3, frameon=False, fontsize=7.5)
    save(fig, f"{case}_3d")


def quantitative():
    R_ = {}
    for m in ["MUSE", "UNet-MD", "HeMIS", "LateMean", "TMC"]:
        f = f"results/mri3d_runs/{m}_s0/test.json"  # tracked copy of every model's test results
        if os.path.exists(f):
            R_[m] = json.load(open(f))
    common = sorted(set.intersection(*[set(v) for v in R_.values()]))
    pats = sorted(R_["MUSE"][common[0]].keys(), key=lambda p: (-p.count("+"), p))
    colours = {"MUSE": "#2a78d6", "UNet-MD": "#eb6834", "HeMIS": "#1baf7a", "LateMean": "#eda100", "TMC": "#e87ba4"}
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.8 / len(R_)
    for i, (m, d) in enumerate(R_.items()):
        vals = [np.mean([np.mean([d[c][p][f"dice_{r}"] for r in ("WT", "TC", "ET")]) for c in common]) for p in pats]
        ax.bar(np.arange(len(pats)) + (i - (len(R_) - 1) / 2) * w, vals, w * 0.92, color=colours[m], label=m, zorder=3)
    short = {"FLAIR": "F", "T1": "T1", "T1gd": "T1c", "T2": "T2"}
    ax.set_xticks(range(len(pats)))
    ax.set_xticklabels(["+".join(short[s] for s in p.split("+")) for p in pats], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"mean Dice (WT, TC, ET)\n{len(common)} test patients")
    ax.set_ylim(0, 1); ax.grid(axis="y", color="#e4e3df", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=len(R_), fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "dice_per_pattern")
    by_k = {}
    for c in common:
        for p, v in R_["MUSE"][c].items():
            by_k.setdefault(p.count("+") + 1, []).append(v["u_mean"])
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ks = sorted(by_k)
    ax.boxplot([by_k[k] for k in ks], positions=ks, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#c9dcf3", edgecolor="#2a78d6"), medianprops=dict(color="#2a78d6", lw=1.6),
               flierprops=dict(marker="o", ms=2.5))
    ax.set_xlabel("number of available sequences"); ax.set_ylabel("MUSE mean uncertainty (brain)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "uncertainty_vs_sequences")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    quantitative()
    split = json.load(open(f"{DATA}/split.json"))["test"]
    if a.cases:
        cases = a.cases.split(",")
    else:  # patients with all three sub-regions and no large intensity artefact, largest tumours first
        sizes = []
        for c in split:
            y, _ = load_full_labels(DATA, c)
            if not all((y == k).sum() > 2000 for k in (1, 2, 3)):
                continue
            x, _, b = load_case(DATA, c)
            _, counts = np.unique(x[0][b], return_counts=True)
            if counts.max() / b.sum() > 0.01:  # skip the few scans with a constant-filled block (data artefact)
                continue
            sizes.append(((y > 0).sum(), c))
        cases = [c for _, c in sorted(sizes, reverse=True)[: a.n]]
    muse, hemis = load_model("MUSE"), load_model("HeMIS")
    for case in cases:
        x, _, _ = load_case(DATA, case)
        y, brain = load_full_labels(DATA, case)
        pm = missing_panel(case, x, y, brain, muse, hemis)
        triplanar(case, x, y, brain, pm)
        render3d(case, y, pm)
        print("figures for", case, flush=True)


if __name__ == "__main__":
    main()

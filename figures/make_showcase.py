"""Showcase figures: every MRI model on the same held-out patients, in the style of make_mri3d_figures.py.

  figures/showcase/<case>_all_models_2d.png    axial slice, rows = sequence patterns, columns = expert, MUSE,
                                               its ablations and the six baselines (mean Dice per tile)
  figures/showcase/<case>_all_models_3d.png    same grid as shaded 3D tumour surfaces
  figures/showcase/<case>_errors_vs_uncertainty.png  where each model is wrong vs where it says it is unsure
  figures/showcase/<case>_dominant_sequence_3d.png   MUSE tumour surface coloured by the sequence that drives it
  figures/showcase/<case>_uncertainty_growth_3d.png  MUSE uncertainty surface as sequences are removed
  figures/showcase/risk_coverage.png           error left after flagging the most uncertain voxels (all cases)

Predictions are cached in figures/showcase/cache/<case>_<model>.npz.
usage: python figures/make_showcase.py [--cases BRATS_448,BRATS_261,BRATS_240]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import figures.make_mri3d_figures as M  # noqa: E402
from muse3d.data import SEQUENCES, load_case, load_full_labels  # noqa: E402

OUT = "figures/showcase"
PATTERNS = M.PATTERNS  # all four, T1gd absent, FLAIR + T2, FLAIR only
SHOW = ["all four", "T1gd absent", "FLAIR only"]  # rows of the all-model grids
MODELS = ["MUSE", "MUSE-noGate", "MUSE-Dempster", "mmFormer", "RFNet", "HeMIS", "TMC", "LateMean", "UNet-MD"]
LABEL = {"mmFormer": "mmFormer-style", "RFNet": "RFNet-style", "MUSE-noGate": "MUSE w/o gate",
         "MUSE-Dempster": "MUSE + Dempster"}
MUSE_C, INK, MUTED = "#2a78d6", "#1b1f3b", "#6b6f78"
ERR = ListedColormap([(0, 0, 0, 0), "#e5534b", "#3d8bfd"])  # correct, false positive, false negative
lab = lambda m: LABEL.get(m, m)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def predictions(case, x, name):
    """{pattern name: (labels uint8, uncertainty float16, dominant sequence uint8 | None)}."""
    os.makedirs(f"{OUT}/cache", exist_ok=True)
    f = f"{OUT}/cache/{case}_{name}.npz"
    if not os.path.exists(f):
        pred = M.predict(M.load_model(name), x, PATTERNS.values())
        arrs = {}
        for i, p in enumerate(PATTERNS.values()):
            labels, unc, contrib = pred[p]
            arrs[f"lab{i}"] = labels.astype(np.uint8)
            arrs[f"unc{i}"] = unc.astype(np.float16)
            if contrib is not None and np.ndim(contrib) == 4:
                arrs[f"dom{i}"] = np.asarray(contrib).argmax(0).astype(np.uint8)
        np.savez_compressed(f, **arrs)
        print("predicted", case, name, flush=True)
    d = np.load(f)
    return {n: (d[f"lab{i}"], d[f"unc{i}"].astype(np.float32), d[f"dom{i}"] if f"dom{i}" in d else None)
            for i, n in enumerate(PATTERNS)}


def mean_dice(pred, y):
    out = []
    for r in (lambda l: l > 0, lambda l: (l == 2) | (l == 3), lambda l: l == 3):
        a, b = r(pred), r(y)
        s = a.sum() + b.sum()
        out.append(1.0 if s == 0 else 2 * (a & b).sum() / s)
    return float(np.mean(out))


def highlight(ax, name):
    for s in ax.spines.values():
        s.set_visible(name == "MUSE")
        s.set_color(MUSE_C); s.set_linewidth(2.2)


def tile_label(ax, text, is_muse, y=0.04):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=7,
            color="white", fontweight="bold" if is_muse else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc=MUSE_C if is_muse else "#00000099", ec="none"))


def legend_regions(fig, extra=()):
    fig.legend(handles=[Patch(color="#2ca02c", label="edema"), Patch(color="#d62728", label="tumour core"),
                        Patch(color="#ffd21f", label="enhancing tumour"), *extra],
               loc="lower center", ncol=3 + len(extra), frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))


def all_models_2d(case, x, y, P, sl):
    cols = ["input", "ground truth"] + MODELS
    fig, axes = plt.subplots(len(SHOW), len(cols), figsize=(1.45 * len(cols), 1.6 * len(SHOW) + 0.6))
    for r, pname in enumerate(SHOW):
        p = PATTERNS[pname]
        bg = x[2] if 2 in p else x[0]  # show a sequence that was actually available
        M.show(axes[r, 0], bg[..., sl], title=cols[0] if r == 0 else None)
        tile_label(axes[r, 0], "T1gd" if 2 in p else "FLAIR", False)
        M.show(axes[r, 1], bg[..., sl], np.where(y > 0, y, 0)[..., sl], M.SEG, title=cols[1] if r == 0 else None)
        for c, m in enumerate(MODELS, start=2):
            pred = P[m][pname][0]
            M.show(axes[r, c], bg[..., sl], np.where(pred > 0, pred, 0)[..., sl], M.SEG,
                   title=lab(m) if r == 0 else None)
            tile_label(axes[r, c], f"Dice {mean_dice(pred, y):.2f}", m == "MUSE")
            highlight(axes[r, c], m)
        axes[r, 0].set_ylabel(pname, fontsize=8.5)
    axes[0, 2].title.set_color(MUSE_C); axes[0, 2].title.set_fontweight("bold")
    legend_regions(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.2, h_pad=0.3)
    save(fig, f"{case}_all_models_2d")


def signed_distance(mask):
    """mm from the mask surface: positive outside, negative inside (1 mm voxels)."""
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~mask) - distance_transform_edt(mask)


def all_models_3d(case, y, P):
    """Expert regions, then each model's whole-tumour surface coloured by its distance to the expert surface."""
    cols = ["ground truth"] + MODELS
    sd = signed_distance(y > 0)
    lim = 5.0
    fig = plt.figure(figsize=(1.75 * len(cols), 1.95 * len(SHOW) + 0.7))
    t = np.argwhere(y > 0)
    lo, hi = t.min(0), t.max(0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 0.72
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes
    cmap = plt.cm.RdBu_r
    for r, pname in enumerate(SHOW):
        for c, m in enumerate(cols):
            ax = fig.add_subplot(len(SHOW), len(cols), r * len(cols) + c + 1, projection="3d")
            if m == "ground truth":
                M.mesh(ax, y > 0, "#2ca02c", alpha=0.16)
                M.mesh(ax, (y == 2) | (y == 3), "#d62728", alpha=0.6)
                M.mesh(ax, y == 3, "#ffd21f", alpha=0.95)
            else:
                pred = P[m][pname][0] > 0
                verts, faces, _, _ = marching_cubes(pred.astype(np.float32), 0.5, step_size=2)
                idx = np.clip(np.round(verts).astype(int), 0, np.array(y.shape) - 1)
                d = sd[tuple(idx.T)]
                tri = verts[faces]
                n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
                shade = 0.75 + 0.25 * np.clip(n @ np.array([0.4, -0.5, 0.75]) / 1.05, -1, 1)
                rgb = cmap(np.clip(d[faces].mean(1) / (2 * lim) + 0.5, 0, 1))[:, :3] * shade[:, None]
                pc = Poly3DCollection(tri, linewidths=0)
                pc.set_facecolor(np.c_[rgb, np.ones(len(faces))])
                ax.add_collection3d(pc)
                is_m = m == "MUSE"
                ax.text2D(0.5, 0.0, f"|d| {np.abs(d).mean():.1f} mm · Dice {mean_dice(P[m][pname][0], y):.2f}",
                          transform=ax.transAxes, ha="center", fontsize=7, color=MUSE_C if is_m else MUTED,
                          fontweight="bold" if is_m else "normal")
            ax.set_xlim(mid[0] - half, mid[0] + half); ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_zlim(mid[2] - half, mid[2] + half)
            ax.view_init(elev=22, azim=-55); ax.set_axis_off()
            is_m = m == "MUSE"
            if r == 0:
                ax.set_title(lab(m), fontsize=8, color=MUSE_C if is_m else INK, fontweight="bold" if is_m else "normal")
            if c == 0:
                ax.text2D(-0.08, 0.5, pname, transform=ax.transAxes, rotation=90, va="center", fontsize=8.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
    cb = fig.colorbar(sm, ax=fig.axes, fraction=0.008, pad=0.03)
    cb.set_label("surface error (mm)\nblue = inside expert, red = outside")
    legend_regions(fig)
    fig.subplots_adjust(left=0.02, right=0.92, top=0.92, bottom=0.07, wspace=-0.05, hspace=0.08)
    save(fig, f"{case}_all_models_3d")


def errors_vs_uncertainty(case, x, y, brain, P, sl):
    """Per model: wrong voxels (FP red / FN blue) next to its uncertainty with the error outline."""
    show_models = ["MUSE", "mmFormer", "HeMIS"]
    fig, axes = plt.subplots(len(SHOW), 2 * len(show_models), figsize=(2.0 * 2 * len(show_models), 1.95 * len(SHOW) + 0.6))
    gt = (y > 0)[..., sl]
    for r, pname in enumerate(SHOW):
        bg = (x[2] if 2 in PATTERNS[pname] else x[0])[..., sl]
        for j, m in enumerate(show_models):
            pred, unc, _ = P[m][pname]
            pr = (pred > 0)[..., sl]
            err = np.where(pr & ~gt, 1, np.where(~pr & gt, 2, 0)).astype(float)
            u = np.where(brain[..., sl], unc[..., sl], np.nan)
            u = u / np.nanpercentile(u, 99.5)  # each model's own scale: what matters is where it is high
            a, b = axes[r, 2 * j], axes[r, 2 * j + 1]
            M.show(a, bg, err, ERR, title=f"{lab(m)}: errors" if r == 0 else None)
            M.show(b, bg, u, "magma", vmax=1.0, title=f"{lab(m)}: uncertainty" if r == 0 else None)
            b.contour(np.rot90(err > 0), levels=[0.5], colors="#7fffd4", linewidths=0.6)
            # how well uncertainty ranks this slice's wrong voxels (AUROC within the brain)
            from sklearn.metrics import roc_auc_score
            inb = brain[..., sl]
            e = (err > 0)[inb]
            if 0 < e.sum() < e.size:
                tile_label(b, f"AUROC {roc_auc_score(e, np.nan_to_num(u[inb])):.2f}", m == "MUSE")
            highlight(a, m); highlight(b, m)
        axes[r, 0].set_ylabel(pname, fontsize=8.5)
    fig.legend(handles=[Patch(color="#e5534b", label="false positive"), Patch(color="#3d8bfd", label="false negative"),
                        Patch(color="#7fffd4", label="error outline on uncertainty map")],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.2, h_pad=0.3)
    save(fig, f"{case}_errors_vs_uncertainty")


def dominant_sequence_3d(case, y, P):
    """All four sequences available: each predicted sub-region coloured by the sequence that drives it."""
    pred, _, dom = P["MUSE"]["all four"]
    regions = {"whole tumour (edema shell)": pred > 0, "tumour core": (pred == 2) | (pred == 3),
               "enhancing tumour": pred == 3}
    fig = plt.figure(figsize=(12, 3.8))
    for i, (rname, region) in enumerate(regions.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        verts = None
        share = []
        for k in range(4):
            v = M.mesh(ax, region & (dom == k), M.SEQ_COL[k], alpha=0.9)
            verts = v if verts is None else (verts if v is None else np.vstack([verts, v]))
            share.append((region & (dom == k)).sum() / max(region.sum(), 1))
        top = int(np.argmax(share))
        M.frame(ax, verts, f"{rname}\n{SEQUENCES[top]} drives {share[top] * 100:.0f}% of voxels")
    fig.legend(handles=[Patch(color=M.SEQ_COL[k], label=f"{SEQUENCES[k]} dominant") for k in range(4)],
               loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("MUSE, all four sequences: which sequence supplies most of the evidence", y=1.02, fontsize=10, color=INK)
    fig.subplots_adjust(left=0, right=1, wspace=0, bottom=0.08)
    save(fig, f"{case}_dominant_sequence_3d")


def uncertainty_growth_3d(case, P):
    on = np.concatenate([P["MUSE"][n][1][P["MUSE"][n][0] > 0] for n in PATTERNS])
    vmax = float(np.percentile(on, 98))
    fig = plt.figure(figsize=(14, 3.8))
    for i, pname in enumerate(PATTERNS):
        pred, unc, _ = P["MUSE"][pname]
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        mean_u = float(unc[pred > 0].mean())
        M.frame(ax, M.mesh(ax, pred > 0, values=unc, vmax=vmax), f"{pname}\nmean u on tumour {mean_u:.3f}")
    sm = plt.cm.ScalarMappable(cmap="magma"); sm.set_array([0, vmax])
    fig.colorbar(sm, ax=fig.axes, fraction=0.012, pad=0.0, label="MUSE uncertainty u")
    fig.subplots_adjust(left=0, right=0.94, wspace=0)
    fig.suptitle("MUSE becomes less certain exactly as sequences are removed", y=1.08, fontsize=10, color=INK)
    save(fig, f"{case}_uncertainty_growth_3d")


def risk_coverage(store):
    """store[model] = list of (wrong, uncertainty) voxel arrays over cases and patterns."""
    cover = np.linspace(0.5, 1.0, 51)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    area = {}
    for m in MODELS[::-1]:
        if m not in store:
            continue
        wrong = np.concatenate([w for w, _ in store[m]]); u = np.concatenate([v for _, v in store[m]])
        order = np.argsort(u, kind="stable")  # keep the most certain voxels first
        cum = np.cumsum(wrong[order]) / np.arange(1, len(order) + 1)
        risk = [cum[int(c * len(order)) - 1] for c in cover]
        area[m] = float(np.trapezoid(risk, cover) / (cover[-1] - cover[0]))
        is_m = m == "MUSE"
        if m.startswith("MUSE-"):
            continue
        a.plot(cover * 100, np.array(risk) * 100, color=MUSE_C if is_m else "#b4b7bd", lw=2.6 if is_m else 1.1,
               zorder=3 if is_m else 2, label=lab(m) if is_m else None)
    a.set_xlabel("voxels kept (most certain first, %)"); a.set_ylabel("error rate on kept voxels (%)")
    a.set_title("Flag the most uncertain voxels: how much error is left?", loc="left", fontsize=9.5, color=INK)
    a.plot([], [], color="#b4b7bd", lw=1.1, label="baselines")
    a.legend(frameon=False, loc="upper left")
    a.invert_xaxis(); a.grid(color="#eeeeee")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    names = sorted(area, key=area.get)
    b.barh(range(len(names)), [area[n] * 100 for n in names],
           color=[MUSE_C if n == "MUSE" else "#9fc1ea" if n.startswith("MUSE-") else "#c3c5ca" for n in names])
    b.set_yticks(range(len(names)), [lab(n) for n in names])
    for t, n in zip(b.get_yticklabels(), names):
        if n == "MUSE":
            t.set_color(MUSE_C); t.set_fontweight("bold")
    b.invert_yaxis()
    b.set_xlabel("mean error over 50-100% coverage (%, lower is better)")
    b.set_title("Area under risk-coverage", loc="left", fontsize=9.5, color=INK)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "risk_coverage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="BRATS_448,BRATS_261,BRATS_240")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    store = {}
    for i, case in enumerate(a.cases.split(",")):
        x, _, _ = load_case(M.DATA, case)
        y, brain = load_full_labels(M.DATA, case)
        P = {m: predictions(case, x, m) for m in MODELS}
        sl = int(np.argmax((y > 0).sum((0, 1))))
        if i == 0:  # the detailed panels for the first patient
            all_models_2d(case, x, y, P, sl)
            all_models_3d(case, y, P)
            errors_vs_uncertainty(case, x, y, brain, P, sl)
            dominant_sequence_3d(case, y, P)
            uncertainty_growth_3d(case, P)
        idx = np.flatnonzero(brain)
        idx = idx[rng.choice(len(idx), min(len(idx), 400_000), replace=False)]  # same voxels for every model
        for m in MODELS:
            for pname in PATTERNS:
                pred, unc, _ = P[m][pname]
                store.setdefault(m, []).append(((pred.ravel()[idx] != y.ravel()[idx]).astype(np.float32),
                                                unc.ravel()[idx]))
        print("done", case, flush=True)
    risk_coverage(store)


if __name__ == "__main__":
    main()

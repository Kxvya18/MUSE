"""Publication-quality renderings of real held-out patients.

  figures/realistic/<case>_3d_regions.png      glass brain + tumour sub-regions: ground truth vs MUSE
                                               (all four sequences, T1gd absent, FLAIR only)
  figures/realistic/<case>_3d_uncertainty.png  MUSE uncertainty painted on the tumour surface
  figures/realistic/<case>_2d_contours.png     zoomed slices with contour outlines (GT / MUSE / HeMIS)
                                               and a translucent uncertainty overlay, per pattern

Predictions are cached in figures/realistic/cache/<case>.npz.
usage: python figures/render_realistic.py [--cases BRATS_448,BRATS_261]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from muse3d.data import load_case, load_full_labels  # noqa: E402

pv.OFF_SCREEN = True
OUT = "figures/realistic"
DATA = "data/mri3d_test1mm"
PATTERNS = {"all four sequences": (0, 1, 2, 3), "T1gd absent": (0, 1, 3), "FLAIR only": (0,)}
REGION_COL = {"edema": "#3fb950", "core": "#e5534b", "enhancing": "#f2cc60"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def predictions(case, x):
    """Cached MUSE / HeMIS predictions (labels + MUSE uncertainty) at full resolution."""
    os.makedirs(f"{OUT}/cache", exist_ok=True)
    f = f"{OUT}/cache/{case}.npz"
    if os.path.exists(f):
        return dict(np.load(f))
    import figures.make_mri3d_figures as M  # lazy: loads torch + models only when needed
    muse, hemis = M.load_model("MUSE"), M.load_model("HeMIS")
    pm, ph = M.predict(muse, x, PATTERNS.values()), M.predict(hemis, x, PATTERNS.values())
    out = {}
    for i, p in enumerate(PATTERNS.values()):
        out[f"muse_{i}"] = pm[p][0].astype(np.uint8)
        out[f"unc_{i}"] = pm[p][1].astype(np.float16)
        out[f"hemis_{i}"] = ph[p][0].astype(np.uint8)
    np.savez_compressed(f, **out)
    return out


def surface(mask, sigma=1.0, level=0.5):
    """Smooth iso-surface of a binary mask as a PyVista mesh."""
    grid = pv.ImageData(dimensions=mask.shape)
    grid.point_data["v"] = gaussian_filter(mask.astype(np.float32), sigma).ravel(order="F")
    return grid.contour([level], scalars="v")


def add_regions(p, lab, brain_mesh):
    p.add_mesh(brain_mesh, color="#d8dde6", opacity=0.12, smooth_shading=True, specular=0.3)
    for name, mask, op in [("edema", lab > 0, 0.22), ("core", (lab == 2) | (lab == 3), 0.45), ("enhancing", lab == 3, 1.0)]:
        if mask.sum() > 30:
            p.add_mesh(surface(mask, 1.0), color=REGION_COL[name], opacity=op, smooth_shading=True, specular=0.4, specular_power=20)


def camera(p, focus, extent):
    """Oblique view that frames the whole brain, focused between brain and tumour centres."""
    d = np.array([-0.55, -0.75, 0.45])
    d /= np.linalg.norm(d)
    p.camera_position = [tuple(focus + d * extent * 2.1), tuple(focus), (0, 0, 1)]
    p.camera.zoom(1.05)


def view(brain, y):
    b = np.argwhere(brain)
    focus = 0.5 * b.mean(0) + 0.5 * np.argwhere(y > 0).mean(0)
    return focus, float((b.max(0) - b.min(0)).max())


def render_regions(case, y, brain, preds):
    brain_mesh = surface(brain, 2.0)
    focus, extent = view(brain, y)
    titles = ["Expert ground truth"] + [f"MUSE · {t}" for t in PATTERNS]
    labs = [y] + [preds[f"muse_{i}"] for i in range(len(PATTERNS))]
    p = pv.Plotter(shape=(1, 4), off_screen=True, window_size=(2400, 700), border=False)
    for j, (t, lab) in enumerate(zip(titles, labs)):
        p.subplot(0, j)
        p.set_background("white")
        add_regions(p, lab, brain_mesh)
        p.add_text(t, font_size=13, color="#1b1f3b", position="upper_edge")
        camera(p, focus, extent)
    p.screenshot(f"{OUT}/{case}_3d_regions.png")
    p.close()


def render_uncertainty(case, y, brain, preds):
    brain_mesh = surface(brain, 2.0)
    focus, extent = view(brain, y)
    vmax = float(np.percentile(np.concatenate([preds[f"unc_{i}"][preds[f"muse_{i}"] > 0].astype(np.float32)
                                               for i in range(len(PATTERNS))]), 98))
    p = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 720), border=False)
    for j, t in enumerate(PATTERNS):
        p.subplot(0, j)
        p.set_background("white")
        p.add_mesh(brain_mesh, color="#d8dde6", opacity=0.12, smooth_shading=True)
        mesh = surface(preds[f"muse_{j}"] > 0, 1.0)
        pts = np.clip(np.round(mesh.points).astype(int), 0, np.array(y.shape) - 1)
        mesh.point_data["uncertainty"] = preds[f"unc_{j}"].astype(np.float32)[tuple(pts.T)]
        p.add_mesh(mesh, scalars="uncertainty", cmap="magma", clim=[0, vmax], smooth_shading=True, specular=0.3,
                   show_scalar_bar=(j == 2), scalar_bar_args={"title": "u", "color": "#1b1f3b",
                                                               "vertical": True, "title_font_size": 16, "label_font_size": 13})
        p.add_text(f"MUSE uncertainty · {t}", font_size=13, color="#1b1f3b", position="upper_edge")
        camera(p, focus, extent)
    p.screenshot(f"{OUT}/{case}_3d_uncertainty.png")
    p.close()


def contours(ax, lab, styles):
    for name, mask in [("edema", lab > 0), ("core", (lab == 2) | (lab == 3)), ("enhancing", lab == 3)]:
        for c in find_contours(mask.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=REGION_COL[name], **styles)


def render_2d(case, x, y, brain, preds):
    sl = int(np.argmax((y > 0).sum((0, 1))))
    box = np.argwhere((y[..., sl] > 0) | (preds["muse_2"][..., sl] > 0))
    lo, hi = np.maximum(box.min(0) - 18, 0), np.minimum(box.max(0) + 18, y.shape[:2])
    crop = lambda a: np.rot90(a[lo[0]:hi[0], lo[1]:hi[1]])
    t1gd, flair = crop(x[2][..., sl].astype(float)), crop(x[0][..., sl].astype(float))
    vmax = float(np.percentile(preds["unc_2"][..., sl], 99.5))
    cols = ["Expert ground truth", "MUSE", "HeMIS", "MUSE uncertainty"]
    fig, axes = plt.subplots(len(PATTERNS), 4, figsize=(11, 3.0 * len(PATTERNS)))
    for r, (pname, p) in enumerate(PATTERNS.items()):
        bg = t1gd if 2 in p else flair  # show a sequence that was actually available
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(bg, cmap="gray", vmin=-1.5, vmax=4.0, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        contours(axes[r, 0], crop(y[..., sl]), {"lw": 1.8})
        contours(axes[r, 1], crop(preds[f"muse_{r}"][..., sl]), {"lw": 1.8})
        contours(axes[r, 2], crop(preds[f"hemis_{r}"][..., sl]), {"lw": 1.8})
        u = crop(np.where(brain[..., sl], preds[f"unc_{r}"][..., sl].astype(float), np.nan))
        im = axes[r, 3].imshow(u, cmap="magma", vmin=0, vmax=vmax, alpha=0.8, interpolation="bilinear")
        contours(axes[r, 3], crop(y[..., sl]), {"lw": 0.9, "ls": "--", "alpha": 0.9})
        axes[r, 0].set_ylabel(pname + ("\n(T1gd shown)" if 2 in p else "\n(FLAIR shown)"), fontsize=10)
        if r == 0:
            for c, t in enumerate(cols):
                axes[r, c].set_title(t, fontsize=11)
    cb = fig.colorbar(im, ax=axes[:, 3], fraction=0.05, pad=0.02)
    cb.set_label("uncertainty u (dashed = expert outline)")
    handles = [plt.Line2D([], [], color=REGION_COL[k], lw=2.5, label=l) for k, l in
               [("edema", "whole tumour (incl. edema)"), ("core", "tumour core"), ("enhancing", "enhancing tumour")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.45, -0.01))
    fig.subplots_adjust(left=0.07, right=0.88, wspace=0.03, hspace=0.06, bottom=0.05)
    fig.savefig(f"{OUT}/{case}_2d_contours.png", dpi=220, bbox_inches="tight")
    fig.savefig(f"{OUT}/{case}_2d_contours.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="BRATS_448,BRATS_261")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for case in a.cases.split(","):
        x, _, _ = load_case(DATA, case)
        y, brain = load_full_labels(DATA, case)
        preds = predictions(case, x)
        render_regions(case, y, brain, preds)
        render_uncertainty(case, y, brain, preds)
        render_2d(case, x, y, brain, preds)
        print("rendered", case, flush=True)


if __name__ == "__main__":
    main()

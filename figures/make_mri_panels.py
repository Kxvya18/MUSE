"""Qualitative MRI panels: real input sequences, ground truth, MUSE segmentation,
MUSE uncertainty map, a baseline's segmentation, and MUSE per-sequence contributions,
for several missing-sequence patterns on held-out test patients.

usage: python figures/make_mri_panels.py [--cases BRATS_xxx,...] [--baseline GAT-zero] [--seed 100]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_brats import crop_and_normalise  # noqa: E402
from run_brats import load_split, batch, pattern_tensor, MOD  # noqa: E402
from run_graph import build  # noqa: E402
from muse.metrics import outputs_to_prob_unc  # noqa: E402

ROOT, SRC = "data/brats_graphs_c4_ms", "data/Task01_BrainTumour/imagesTr"
# 0 bg, 1 healthy, 2 edema, 3 necrotic/non-enhancing core, 4 enhancing
SEG = ListedColormap([(0, 0, 0, 0), (0, 0, 0, 0), "#2ca02c", "#d62728", "#ffd21f"])
PATTERNS = {"all four": (0, 1, 2, 3), "T1gd absent": (0, 1, 3), "FLAIR + T2": (0, 3), "FLAIR only": (0,)}


def paint(coords, values, shape, c):
    grid = np.zeros(shape, dtype=np.float32)
    grid[tuple(coords.T)] = values
    return grid.repeat(c, 0).repeat(c, 1).repeat(c, 2)


def load_model(name, cfg, seed, dev):
    model, _ = build(name, [23] * 4, 5, **cfg)
    model.load_state_dict(torch.load(f"checkpoints/brats_{name}_{seed}.pt", map_location=dev))
    return model.to(dev).eval()


@torch.no_grad()
def predict(model, g, pattern, dev):
    xs, y, ei, m = batch([g], [pattern_tensor(pattern)], dev)
    out = model(xs, ei, m)
    prob, unc = outputs_to_prob_unc(out)
    contrib = None
    if "contrib" in out:  # propagated discounted evidence per modality: [K, N, C]
        contrib = out["contrib"].sum(-1).cpu().numpy()  # K, N
    elif "gammas" in out:
        contrib = (((out["alphas"] - 1) * out["gammas"]).sum(-1)).cpu().numpy()
    return prob.argmax(-1).cpu().numpy(), unc.cpu().numpy(), contrib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--baseline", default="GAT-zero")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--results", default="results/brats", help="dir with final_<model>_<seed>.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = "cpu"
    split, get = load_split(ROOT)
    cfgs = {}
    for name in ("MUSE", a.baseline):
        r = json.load(open(f"{a.results}/final_{name}_{a.seed}.json"))
        cfgs[name] = r["cfg"]
    muse, base = load_model("MUSE", cfgs["MUSE"], a.seed, dev), load_model(a.baseline, cfgs[a.baseline], a.seed, dev)
    cases = a.cases.split(",") if a.cases else split["test"][: a.n]
    os.makedirs("figures/mri", exist_ok=True)
    for case in cases:
        z = np.load(os.path.join(ROOT, case + ".npz"))
        c, gs, coords = int(z["c"]), z["grid_shape"], z["coords"]
        img = np.asarray(nib.load(os.path.join(SRC, case + ".nii.gz")).dataobj, dtype=np.float32)
        lab = np.asarray(nib.load(os.path.join(SRC.replace("imagesTr", "labelsTr"), case + ".nii.gz")).dataobj).astype(np.uint8)
        img, lab, brain = crop_and_normalise(img, lab, c)
        sl = int(np.argmax(((lab > 0).sum((0, 1)))))  # axial slice with the most tumour
        gt = np.where(brain, 1, 0)
        gt[lab == 1], gt[lab == 2], gt[lab == 3] = 2, 3, 4
        g = get(case)
        rows = len(PATTERNS)
        fig, axes = plt.subplots(rows, 9, figsize=(13.5, 1.65 * rows + 0.5),
                                 gridspec_kw={"width_ratios": [1] * 8 + [1.15]})
        for r, (pname, pat) in enumerate(PATTERNS.items()):
            pred, unc, contrib = predict(muse, g, pat, dev)
            bpred, _, _ = predict(base, g, pat, dev)
            P = paint(coords, pred, gs, c)[: lab.shape[0], : lab.shape[1], : lab.shape[2]]
            U = paint(coords, unc, gs, c)[: lab.shape[0], : lab.shape[1], : lab.shape[2]]
            B = paint(coords, bpred, gs, c)[: lab.shape[0], : lab.shape[1], : lab.shape[2]]
            for k in range(4):
                ax = axes[r, k]
                ax.imshow(np.rot90(img[:, :, sl, k]), cmap="gray", vmin=-2, vmax=4)
                if k not in pat:
                    ax.imshow(np.rot90(np.ones_like(img[:, :, sl, k])), cmap="gray", vmin=0, vmax=1, alpha=0.82)
                    ax.text(0.5, 0.5, "absent", transform=ax.transAxes, ha="center", va="center",
                            fontsize=8, color="#444")
            base_img = np.rot90(img[:, :, sl, 0])
            for j, (vol, title) in enumerate([(gt, "ground truth"), (P, "MUSE"), (B, a.baseline)]):
                ax = axes[r, 4 + j]
                ax.imshow(base_img, cmap="gray", vmin=-2, vmax=4)
                ax.imshow(np.rot90(vol[:, :, sl]), cmap=SEG, vmin=-0.5, vmax=4.5, interpolation="nearest", alpha=0.75)
                if r == 0:
                    ax.set_title(title, fontsize=8.5)
            ax = axes[r, 7]
            ax.imshow(base_img, cmap="gray", vmin=-2, vmax=4)
            ax.imshow(np.rot90(np.where(brain[:, :, sl], U[:, :, sl], np.nan)),
                      cmap="magma", vmin=0, vmax=float(np.nanmax(unc)), alpha=0.85)
            if r == 0:
                ax.set_title("MUSE uncertainty u", fontsize=8.5)
            # contributions: share of fused evidence per sequence on tumour nodes
            ax = axes[r, 8]
            tum = g["y"].numpy() >= 2
            share = contrib[:, tum].sum(1) if contrib is not None else np.zeros(4)
            share = share / max(share.sum(), 1e-9)
            ax.barh(range(4), share[::-1], color=["#4c78a8" if (3 - i) in pat else "#cccccc" for i in range(4)])
            ax.set_yticks(range(4)); ax.set_yticklabels(MOD[::-1], fontsize=7)
            ax.set_xlim(0, 1); ax.tick_params(axis="x", labelsize=6.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title("contribution", fontsize=8.5)
            axes[r, 0].set_ylabel(pname, fontsize=8.5)
            for k in range(8):
                axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
                if r == 0 and k < 4:
                    axes[r, k].set_title(MOD[k], fontsize=8.5)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color="#2ca02c", label="edema"), Patch(color="#d62728", label="necrotic / non-enh. core"),
                            Patch(color="#ffd21f", label="enhancing tumour")],
                   loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.45, -0.01))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for ext in ("pdf", "png"):
            fig.savefig(f"figures/mri/{case}_panel{a.tag}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote", case)


if __name__ == "__main__":
    main()

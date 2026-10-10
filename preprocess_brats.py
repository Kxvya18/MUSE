"""MSD Task01 BrainTumour -> per-patient supervoxel graphs for MUSE.

Graph construction is deliberately *modality-agnostic*: nodes are regular c x c x c
cubes over the brain bounding box (no intensity-driven clustering, which would leak
information from modalities that may be missing at inference), edges are the
6-connected cube lattice. Each modality (FLAIR, T1, T1gd, T2) gets its own node
feature vector computed only from that modality's voxels in the cube:
  mean, std, min, max, 10/25/50/75/90th percentiles, and the 8 octant means (17 dims),
after per-patient z-scoring inside the brain mask.

Node classes (frame of discernment, 5 classes):
  0 background (cube mostly outside brain), 1 healthy brain, 2 edema,
  3 necrotic / non-enhancing core, 4 enhancing tumour  (majority vote in the cube).
MSD label map: 0 bg, 1 edema, 2 non-enhancing tumour, 3 enhancing tumour.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np

SRC = "data/Task01_BrainTumour"
MODALITIES = ["FLAIR", "T1", "T1gd", "T2"]  # MSD channel order


def cube_view(vol, c):
    """[X,Y,Z] (multiple of c) -> [nx,ny,nz, c,c,c]"""
    X, Y, Z = vol.shape
    v = vol.reshape(X // c, c, Y // c, c, Z // c, c)
    return v.transpose(0, 2, 4, 1, 3, 5)


def crop_and_normalise(img, lab, c):
    """Crop to the brain bounding box padded to a multiple of c; z-score each modality
    inside the brain. Shared with the figure scripts so images align with node grids."""
    brain = img.max(-1) > 0
    idx = np.argwhere(brain)
    lo, hi = idx.min(0), idx.max(0) + 1
    hi = hi + (-(hi - lo)) % c
    shift = np.maximum(hi - np.array(img.shape[:3]), 0)
    lo, hi = lo - shift, hi - shift
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    img, lab, brain = img[sl].copy(), lab[sl], brain[sl]
    for m in range(4):
        v = img[..., m][brain]
        img[..., m] = np.where(brain, (img[..., m] - v.mean()) / (v.std() + 1e-6), 0)
    return img, lab, brain


def process(args):
    path, c, out = args
    case = os.path.basename(path).replace(".nii.gz", "")
    f = os.path.join(out, case + ".npz")
    if os.path.exists(f):
        return case
    img = np.asarray(nib.load(path).dataobj, dtype=np.float32)  # X,Y,Z,4
    lab = np.asarray(nib.load(path.replace("imagesTr", "labelsTr")).dataobj).astype(np.uint8)
    img, lab, brain = crop_and_normalise(img, lab, c)
    # node labels: 5 classes
    lab5 = np.where(brain, 1, 0).astype(np.uint8)
    lab5[lab == 1] = 2
    lab5[lab == 2] = 3
    lab5[lab == 3] = 4
    L = cube_view(lab5, c).reshape(*(np.array(lab5.shape) // c), -1)
    counts = np.stack([(L == k).sum(-1) for k in range(5)], -1)
    # majority over tissue classes; background only if most voxels are outside brain
    tissue = counts[..., 1:]
    y = np.where(counts[..., 0] > c ** 3 / 2, 0, tissue.argmax(-1) + 1)
    # keep cubes that touch the brain
    keep = counts[..., 0] < c ** 3
    grid = np.full(keep.shape, -1, np.int64)
    grid[keep] = np.arange(keep.sum())
    feats = []
    for m in range(4):
        V = cube_view(img[..., m], c)[keep]  # n, c,c,c
        flat = V.reshape(len(V), -1)
        h = c // 2
        octs = [V[:, a:a + h, b:b + h, d:d + h].reshape(len(V), -1).mean(1)
                for a in (0, h) for b in (0, h) for d in (0, h)]
        q = np.percentile(flat, [10, 25, 50, 75, 90], axis=1).T
        feats.append(np.concatenate([flat.mean(1, keepdims=True), flat.std(1, keepdims=True),
                                     flat.min(1, keepdims=True), flat.max(1, keepdims=True), q,
                                     np.stack(octs, 1)], 1))
    feats = np.stack(feats, 1).astype(np.float32)  # n, 4, 17
    # 6-connected lattice edges (both directions)
    edges = []
    for ax in range(3):
        a = grid
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1); s2[ax] = slice(1, None)
        u, v = a[tuple(s1)].ravel(), a[tuple(s2)].ravel()
        ok = (u >= 0) & (v >= 0)
        edges.append(np.stack([u[ok], v[ok]]))
    e = np.concatenate(edges, 1)
    e = np.concatenate([e, e[::-1]], 1)
    coords = np.argwhere(keep).astype(np.int16)
    np.savez_compressed(f, x=feats, y=y[keep].astype(np.uint8), edge_index=e.astype(np.int32),
                        coords=coords, grid_shape=np.array(keep.shape), label=lab, c=c)
    return case


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = f"data/brats_graphs_c{a.cube}"
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "imagesTr", "BRATS_*.nii.gz")))
    if a.limit:
        files = files[:a.limit]
    with Pool(a.workers) as p:
        for i, case in enumerate(p.imap_unordered(process, [(f, a.cube, out) for f in files])):
            if i % 50 == 0:
                print(i, case, flush=True)
    # fixed patient-level split 70/10/20 (seed 0)
    cases = sorted(os.path.basename(f).replace(".nii.gz", "") for f in files)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(cases))
    n_tr, n_va = int(0.7 * len(cases)), int(0.1 * len(cases))
    split = {"train": [cases[i] for i in perm[:n_tr]], "val": [cases[i] for i in perm[n_tr:n_tr + n_va]],
             "test": [cases[i] for i in perm[n_tr + n_va:]]}
    json.dump(split, open(os.path.join(out, "split.json"), "w"), indent=1)
    print("done", {k: len(v) for k, v in split.items()})

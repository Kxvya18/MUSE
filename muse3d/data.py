"""MSD Task01 brain-tumour MRI for voxel-level MUSE.

Preprocessing (once per machine): crop each scan to its brain bounding box, z-score each
sequence inside the brain, store as float16 (4, X, Y, Z) + uint8 labels in one .npz per case.

Labels (MSD): 0 background, 1 edema, 2 non-enhancing tumour, 3 enhancing tumour.
Evaluation regions: WT = {1,2,3}, TC = {2,3}, ET = {3}.
Sequences (MSD channel order): FLAIR, T1, T1gd, T2.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import torch

SEQUENCES = ["FLAIR", "T1", "T1gd", "T2"]
PATTERNS = [p for r in range(1, 5) for p in itertools.combinations(range(4), r)]  # all 15


def preprocess_case(args):
    src, out_dir = args
    case = os.path.basename(src).replace(".nii.gz", "")
    out = os.path.join(out_dir, case + ".npz")
    if os.path.exists(out):
        return case
    img = np.asarray(nib.load(src).dataobj, dtype=np.float32)  # X, Y, Z, 4
    lab = np.asarray(nib.load(src.replace("imagesTr", "labelsTr")).dataobj).astype(np.uint8)
    brain = img.max(-1) > 0
    idx = np.argwhere(brain)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    img, lab, brain = img[sl], lab[sl], brain[sl]
    for k in range(4):
        v = img[..., k][brain]
        img[..., k] = np.where(brain, (img[..., k] - v.mean()) / (v.std() + 1e-6), 0)
    np.savez_compressed(out, x=img.transpose(3, 0, 1, 2).astype(np.float16), y=lab, brain=brain)
    return case


def preprocess_all(src_root, out_dir, workers=2):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_root, "imagesTr", "BRATS_*.nii.gz")))
    with Pool(workers) as p:
        for i, _ in enumerate(p.imap_unordered(preprocess_case, [(f, out_dir) for f in files])):
            if i % 50 == 0:
                print(f"preprocessed {i}/{len(files)}", flush=True)
    # same patient-level split as the graph experiments (70/10/20, seed 0)
    cases = sorted(os.path.basename(f).replace(".nii.gz", "") for f in files)
    perm = np.random.default_rng(0).permutation(len(cases))
    n_tr, n_va = int(0.7 * len(cases)), int(0.1 * len(cases))
    split = {"train": [cases[i] for i in perm[:n_tr]], "val": [cases[i] for i in perm[n_tr:n_tr + n_va]],
             "test": [cases[i] for i in perm[n_tr + n_va:]]}
    json.dump(split, open(os.path.join(out_dir, "split.json"), "w"), indent=1)
    return split


def load_case(data_dir, case):
    z = np.load(os.path.join(data_dir, case + ".npz"))
    return z["x"], z["y"], z["brain"]


def random_crop(x, y, size, rng, p_tumour=0.67):
    """Random size^3 crop; centred on a tumour voxel with probability p_tumour. Pads if needed."""
    pad = [(0, 0)] + [(0, max(0, size - s)) for s in y.shape]
    if any(p[1] for p in pad[1:]):
        x, y = np.pad(x, pad), np.pad(y, pad[1:])
    tumour = np.argwhere(y > 0)
    if len(tumour) and rng.random() < p_tumour:
        c = tumour[rng.integers(len(tumour))]
    else:
        c = np.array([rng.integers(s) for s in y.shape])
    lo = [int(np.clip(ci - size // 2, 0, s - size)) for ci, s in zip(c, y.shape)]
    sl = tuple(slice(l, l + size) for l in lo)
    return x[(slice(None),) + sl], y[sl]


class TrainSet(torch.utils.data.Dataset):
    """Random tumour-biased crops with flips; each sample draws one of the 15 sequence patterns."""

    def __init__(self, data_dir, cases, size=96):
        self.data_dir, self.cases, self.size = data_dir, cases, size

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        # torch's RNG is seeded differently in every dataloader worker
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        x, y, _ = load_case(self.data_dir, self.cases[i])
        x, y = random_crop(x.astype(np.float32), y, self.size, rng)
        for ax in range(3):
            if rng.random() < 0.5:
                x, y = np.flip(x, ax + 1), np.flip(y, ax)
        mask = np.zeros(4, dtype=bool)
        mask[list(PATTERNS[rng.integers(len(PATTERNS))])] = True
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()).long(), torch.from_numpy(mask)

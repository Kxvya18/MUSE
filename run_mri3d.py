"""Voxel-level brain-tumour MRI experiment (MSD Task01) for MUSE and baselines.

    python run_mri3d.py prep  --src data/Task01_BrainTumour --data data/mri3d
    python run_mri3d.py train --model MUSE --seed 0 --data data/mri3d --out runs
    python run_mri3d.py test  --model MUSE --seed 0 --data data/mri3d --out runs

Protocol (identical for every model): random 96^3 tumour-biased crops, one of the 15
sequence patterns drawn per sample, AdamW + cosine schedule, mixed precision. Training
resumes from runs/<model>_s<seed>/last.pt. The best checkpoint is chosen on validation
patients (mean Dice WT/TC/ET over a fixed random pattern per patient). Test: all 98
held-out patients x all 15 patterns, sliding-window inference.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from muse3d.data import PATTERNS, SEQUENCES, TrainSet, load_case, preprocess_all
from muse3d.losses import loss_for
from muse3d.models import MODELS, per_sequence_features, probs_and_uncertainty

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
REGIONS = {"WT": (1, 2, 3), "TC": (2, 3), "ET": (3,)}


# ----------------------------------------------------------------- inference
def windows(shape, patch, stride):
    starts = []
    for s in shape:
        last = max(s - patch, 0)
        starts.append(sorted(set(list(range(0, last + 1, stride)) + [last])))
    return [(a, b, c) for a in starts[0] for b in starts[1] for c in starts[2]]


@torch.no_grad()
def predict_case(model, x, patterns, patch=128, stride=96):
    """Sliding-window prediction for several sequence patterns at once.

    Per-sequence models encode every window once and re-fuse per pattern; UNet-MD (early
    fusion) needs one pass per pattern. Returns {pattern: (prob [C,...], unc [...], contrib)}.
    """
    model.eval()
    C = 4
    shape = x.shape[1:]
    pad = [(0, max(0, patch - s)) for s in shape]
    xp = np.pad(x, [(0, 0)] + pad)
    acc = {p: [np.zeros((C,) + xp.shape[1:], np.float32), np.zeros(xp.shape[1:], np.float32),
               np.zeros((4,) + xp.shape[1:], np.float32)] for p in patterns}
    count = np.zeros(xp.shape[1:], np.float32)
    for (a, b, c) in windows(xp.shape[1:], patch, stride):
        sl = (slice(a, a + patch), slice(b, b + patch), slice(c, c + patch))
        xt = torch.from_numpy(xp[(slice(None),) + sl].astype(np.float32))[None].to(DEV)
        with torch.autocast(DEV, enabled=DEV == "cuda"):
            H = None
            if hasattr(model, "fuse"):
                full = torch.ones(1, 4, dtype=torch.bool, device=DEV)
                H = per_sequence_features(model.encoders, xt, full, model.F)
            for p in patterns:
                mask = torch.zeros(1, 4, dtype=torch.bool, device=DEV)
                mask[0, list(p)] = True
                out = model.fuse(H, mask) if H is not None else model(xt, mask)
                prob, unc = probs_and_uncertainty(out)
                acc[p][0][(slice(None),) + sl] += prob[0].float().cpu().numpy()
                acc[p][1][sl] += unc[0].float().cpu().numpy()
                if "gammas" in out:  # evidence contributed by each sequence (discounted excess)
                    contrib = ((out["alphas"] - 1) * out["gammas"]).sum(2)[:, 0]
                    acc[p][2][(slice(None),) + sl] += contrib.float().cpu().numpy()
        count[sl] += 1
    crop = tuple(slice(0, s) for s in shape)
    res = {}
    for p in patterns:
        prob = acc[p][0][(slice(None),) + crop] / count[crop]
        unc = acc[p][1][crop] / count[crop]
        contrib = acc[p][2][(slice(None),) + crop] / count[crop]
        res[p] = (prob, unc, contrib)
    return res


# ----------------------------------------------------------------- metrics
def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def hd95(a, b):
    """95th-percentile symmetric surface distance (voxels = mm at 1 mm spacing)."""
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return 373.13  # BraTS convention: diagonal of the volume when one mask is empty
    box = np.argwhere(a | b)
    lo, hi = np.maximum(box.min(0) - 2, 0), box.max(0) + 3
    a, b = a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], b[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sa, sb = a ^ ndimage.binary_erosion(a), b ^ ndimage.binary_erosion(b)
    da, db = ndimage.distance_transform_edt(~sb)[sa], ndimage.distance_transform_edt(~sa)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def ece(conf, correct, bins=15):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(correct[s].mean() - conf[s].mean())
    return float(e)


def case_metrics(prob, unc, contrib, y, brain, rng):
    pred = prob.argmax(0)
    m = {}
    for r, labels in REGIONS.items():
        P, G = np.isin(pred, labels), np.isin(y, labels)
        m[f"dice_{r}"], m[f"hd95_{r}"] = dice(P, G), hd95(P, G)
    vox = np.argwhere(brain)
    vox = vox[rng.choice(len(vox), min(len(vox), 200_000), replace=False)]
    idx = tuple(vox.T)
    conf, correct, u = prob.max(0)[idx], (pred[idx] == y[idx]), unc[idx]
    m["ece"] = ece(conf, correct.astype(float))
    m["auroc_err"] = float(roc_auc_score(~correct, u)) if 0 < correct.sum() < len(correct) else float("nan")
    # uncertainty at the tumour boundary vs inside the tumour / healthy brain
    wt = y > 0
    edge = wt ^ ndimage.binary_erosion(wt) if wt.any() else wt
    m["u_boundary"] = float(unc[edge].mean()) if edge.any() else float("nan")
    m["u_tumour_core"] = float(unc[ndimage.binary_erosion(wt, iterations=3)].mean()) if wt.sum() > 500 else float("nan")
    m["u_healthy"] = float(unc[brain & ~ndimage.binary_dilation(wt, iterations=3)].mean())
    m["u_mean"] = float(unc[brain].mean())
    if contrib.any():
        share = contrib[:, wt].sum(1) if wt.any() else contrib.reshape(4, -1).sum(1)
        m["contrib"] = (share / max(share.sum(), 1e-9)).tolist()
    return m, pred


# ----------------------------------------------------------------- train / test
def evaluate_val(model, data_dir, cases, seed):
    rng = np.random.default_rng(seed)
    scores = []
    for case in cases:
        p = PATTERNS[rng.integers(len(PATTERNS))]
        x, y, brain = load_case(data_dir, case)
        prob, _, _ = predict_case(model, x.astype(np.float32), [p])[p]
        pred = prob.argmax(0)
        scores.append(np.mean([dice(np.isin(pred, l), np.isin(y, l)) for l in REGIONS.values()]))
    return float(np.mean(scores))


def train(a):
    split = json.load(open(os.path.join(a.data, "split.json")))
    run = os.path.join(a.out, f"{a.model}_s{a.seed}")
    os.makedirs(run, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = MODELS[a.model]().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    it, best = 0, -1.0
    last = os.path.join(run, "last.pt")
    if os.path.exists(last):  # resume
        ck = torch.load(last, map_location=DEV)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        it, best = ck["it"], ck["best"]
        print(f"resumed {a.model} s{a.seed} at iteration {it} (best val {best:.4f})", flush=True)
    loss_fn = loss_for(a.model)
    loader = torch.utils.data.DataLoader(TrainSet(a.data, split["train"], a.patch), batch_size=a.batch,
                                         shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    t0 = time.time()
    while it < a.iters:
        for x, y, mask in loader:
            model.train()
            x, y, mask = x.to(DEV), y.to(DEV), mask.to(DEV)
            with torch.autocast(DEV, enabled=DEV == "cuda"):
                out = model(x, mask)
            loss = loss_fn(out, y, anneal=min(1.0, it / (0.2 * a.iters)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt); scaler.update(); sched.step()
            it += 1
            if it % 50 == 0:
                print(f"{a.model} s{a.seed} it {it}/{a.iters} loss {loss.item():.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
            if it % a.val_every == 0 or it == a.iters:
                v = evaluate_val(model, a.data, split["val"][: a.val_cases], a.seed)
                if v > best:
                    best = v
                    torch.save(model.state_dict(), os.path.join(run, "best.pt"))
                print(f"  val mean Dice {v:.4f} (best {best:.4f})", flush=True)
            if it % a.save_every == 0 or it == a.iters:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "it": it, "best": best}, last)
            if it >= a.iters:
                break
    json.dump({"model": a.model, "seed": a.seed, "best_val": best, "iters": a.iters}, open(os.path.join(run, "train.json"), "w"))


def test(a):
    split = json.load(open(os.path.join(a.data, "split.json")))
    run = os.path.join(a.out, f"{a.model}_s{a.seed}")
    out_file = os.path.join(run, "test.json")
    done = json.load(open(out_file)) if os.path.exists(out_file) else {}
    model = MODELS[a.model]().to(DEV)
    model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location=DEV))
    rng = np.random.default_rng(0)
    show = {(0, 1, 2, 3), (0, 1, 3), (0, 3), (0,)}
    os.makedirs(os.path.join(run, "slices"), exist_ok=True)
    t0 = time.time()
    for i, case in enumerate(split["test"]):
        if case in done:
            continue
        x, y, brain = load_case(a.data, case)
        res = predict_case(model, x.astype(np.float32), PATTERNS)
        done[case] = {}
        sl = int(np.argmax((y > 0).sum((0, 1))))  # axial slice with most tumour (for figures)
        keep = {}
        for p, (prob, unc, contrib) in res.items():
            m, pred = case_metrics(prob, unc, contrib, y, brain, rng)
            done[case]["+".join(SEQUENCES[k] for k in p)] = m
            if p in show and i < 8:
                tag = "".join(map(str, p))
                keep[f"pred_{tag}"], keep[f"unc_{tag}"] = pred[:, :, sl].astype(np.uint8), unc[:, :, sl].astype(np.float16)
        if keep:
            np.savez_compressed(os.path.join(run, "slices", f"{case}.npz"), slice=sl, **keep)
        json.dump(done, open(out_file, "w"))
        print(f"test {a.model} s{a.seed} {i+1}/{len(split['test'])} ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "train", "test"])
    ap.add_argument("--src", default="data/Task01_BrainTumour")
    ap.add_argument("--data", default="data/mri3d")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--model", default="MUSE", choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_cases", type=int, default=24)
    ap.add_argument("--save_every", type=int, default=250)
    a = ap.parse_args()
    if a.stage == "prep":
        preprocess_all(a.src, a.data, workers=a.workers)
    elif a.stage == "train":
        train(a)
    else:
        test(a)

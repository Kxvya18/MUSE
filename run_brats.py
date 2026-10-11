"""MUSE on brain-tumour MRI (MSD Task01 / BraTS-derived), supervoxel graphs.

Protocol (identical for all models):
  * missingness is per *patient* (a sequence is absent for the whole scan); during
    training each patient draws one of the 15 non-empty patterns uniformly;
  * class imbalance: every step trains on all tumour nodes plus an equal-size random
    subset of background/healthy nodes (same sampler for every model/loss);
  * model selection on validation patients (mean node macro-F1 over the 15 patterns);
  * test on held-out patients for all 15 patterns: voxel Dice (WT/TC/ET),
    node accuracy / macro-F1 / ECE / NLL, error-detection AUROC of the uncertainty,
    and boundary-uncertainty statistics.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from muse.metrics import evaluate, outputs_to_prob_unc
from run_graph import build, MAIN, ABLATIONS

K = 4
PATTERNS = [p for r in range(1, K + 1) for p in itertools.combinations(range(K), r)]
MOD = ["FLAIR", "T1", "T1gd", "T2"]


def load_split(root):
    split = json.load(open(os.path.join(root, "split.json")))
    cache = {}

    def get(case):
        if case not in cache:
            z = np.load(os.path.join(root, case + ".npz"))
            cache[case] = {"x": torch.from_numpy(z["x"].astype(np.float16)), "y": torch.from_numpy(z["y"]),
                           "ei": torch.from_numpy(z["edge_index"]), "coords": torch.from_numpy(z["coords"])}
        g = cache[case]  # compact storage (fp16 / int32 / uint8 / int16); expand on use
        return {"x": g["x"].float(), "y": g["y"].long(), "ei": g["ei"].long(), "coords": g["coords"].long()}

    return split, get


def crop(g, rng, size=24, p_tumour=0.7):
    """Random size^3-cube patch (96 mm at c=4), centred on a tumour node w.p. p_tumour."""
    from torch_geometric.utils import subgraph
    tum = (g["y"] >= 2).nonzero(as_tuple=True)[0]
    pool = tum if (tum.numel() and rng.random() < p_tumour) else (g["y"] >= 1).nonzero(as_tuple=True)[0]
    c = g["coords"][pool[rng.integers(pool.numel())]]
    lo = c - size // 2
    keep = ((g["coords"] >= lo) & (g["coords"] < lo + size)).all(1)
    idx = keep.nonzero(as_tuple=True)[0]
    ei, _ = subgraph(idx, g["ei"], relabel_nodes=True, num_nodes=g["y"].numel())
    return {"x": g["x"][idx], "y": g["y"][idx], "ei": ei}


def batch(graphs, masks, dev):
    xs, ys, eis, ms, off = [[] for _ in range(K)], [], [], [], 0
    for g, m in zip(graphs, masks):
        n = g["y"].numel()
        for k in range(K):
            xs[k].append(g["x"][:, k])
        ys.append(g["y"]); eis.append(g["ei"] + off); ms.append(m.expand(n, K)); off += n
    return ([torch.cat(x).to(dev) for x in xs], torch.cat(ys).to(dev), torch.cat(eis, 1).to(dev),
            torch.cat(ms).to(dev))


def pattern_tensor(p):
    m = torch.zeros(1, K, dtype=torch.bool)
    m[0, list(p)] = True
    return m


def dice(pred, gt):
    inter = (pred & gt).sum()
    s = pred.sum() + gt.sum()
    return float(2 * inter / s) if s > 0 else 1.0


def voxel_dice(case_root, case, node_pred):
    """Paint node labels back to voxels, compare with original MSD labels (WT/TC/ET)."""
    z = np.load(os.path.join(case_root, case + ".npz"))
    c, gs, coords, lab = int(z["c"]), z["grid_shape"], z["coords"], z["label"]
    grid = np.zeros(gs, np.uint8)
    grid[tuple(coords.T)] = node_pred
    vol = grid.repeat(c, 0).repeat(c, 1).repeat(c, 2)[:lab.shape[0], :lab.shape[1], :lab.shape[2]]
    # MSD: 1 edema, 2 non-enh, 3 enh ; ours: 2 edema, 3 core, 4 enh
    return {"WT": dice(vol >= 2, lab >= 1), "TC": dice((vol == 3) | (vol == 4), (lab == 2) | (lab == 3)),
            "ET": dice(vol == 4, lab == 3)}


def train(model_name, cfg, seed, root, epochs=40, B=4, dev="mps", patience=8, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    split, get = load_split(root)
    split_, get_ = load_split(root)
    F_in = get_(split_["train"][0])["x"].size(-1)
    model, loss_fn = build(model_name, [F_in] * K, 5, **cfg)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tr = split["train"]
    rng = np.random.default_rng(seed)
    val_pats = [PATTERNS[i] for i in rng.choice(len(PATTERNS), len(split["val"]))]  # fixed
    best, best_state, bad, step = -1, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), B):
            cases = [tr[j] for j in perm[i:i + B]]
            masks = [pattern_tensor(PATTERNS[rng.integers(len(PATTERNS))]) for _ in cases]
            xs, y, ei, m = batch([crop(get(c), rng) for c in cases], masks, dev)
            tumour = (y >= 2).nonzero(as_tuple=True)[0]
            rest = (y < 2).nonzero(as_tuple=True)[0]
            rest = rest[torch.randperm(rest.numel(), device=dev)[:max(tumour.numel(), 256)]]
            idx = torch.cat([tumour, rest])
            opt.zero_grad()
            loss, _ = loss_fn(model(xs, ei, m), y, idx, anneal=min(1.0, step / 500))
            loss.backward(); opt.step(); step += 1
        v = validate(model, split["val"], val_pats, get, dev)
        if log:
            print(f"  ep{ep} val_f1={v:.4f}", file=log, flush=True)
        if v > best:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


@torch.no_grad()
def validate(model, cases, pats, get, dev):
    model.eval()
    f1s = []
    for c, p in zip(cases, pats):
        xs, y, ei, m = batch([get(c)], [pattern_tensor(p)], dev)
        f1s.append(evaluate(model(xs, ei, m), y, torch.arange(y.numel(), device=dev))["f1"])
    return float(np.mean(f1s))


@torch.no_grad()
def test(model, root, dev="mps"):
    model.eval()
    split, get = load_split(root)
    res = {}
    for p in PATTERNS:
        key = "+".join(MOD[k] for k in p)
        agg = {"dice": [], "node": [], "bnd": []}
        for c in split["test"]:
            g = get(c)
            xs, y, ei, m = batch([g], [pattern_tensor(p)], dev)
            out = model(xs, ei, m)
            idx = torch.arange(y.numel(), device=dev)
            agg["node"].append(evaluate(out, y, idx))
            prob, unc = outputs_to_prob_unc(out)
            pred = prob.argmax(-1).cpu().numpy()
            agg["dice"].append(voxel_dice(root, c, pred))
            # boundary nodes: some 6-neighbour has a different ground-truth label
            yy, e = g["y"].numpy(), g["ei"].numpy()
            bnd = np.zeros(len(yy), bool)
            diff = yy[e[0]] != yy[e[1]]
            bnd[e[0][diff]] = True
            u = unc.cpu().numpy()
            brain = yy > 0
            try:
                au = roc_auc_score(bnd[brain], u[brain])
            except ValueError:
                au = float("nan")
            agg["bnd"].append({"u_boundary": float(u[bnd & brain].mean()) if (bnd & brain).any() else float("nan"),
                               "u_interior": float(u[~bnd & brain].mean()),
                               "auroc_boundary": au,
                               "spearman_u_err": float(spearmanr(u, pred != yy).statistic)})
        res[key] = {
            "dice": {k: float(np.mean([d[k] for d in agg["dice"]])) for k in ("WT", "TC", "ET")},
            "node": {k: float(np.nanmean([d[k] for d in agg["node"]])) for k in agg["node"][0]},
            "boundary": {k: float(np.nanmean([d[k] for d in agg["bnd"]])) for k in agg["bnd"][0]},
            "n_modalities": len(p),
        }
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "final"])
    ap.add_argument("--root", default="data/brats_graphs_c4_ms")
    ap.add_argument("--models", default=",".join(MAIN))
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--cfg", default='{"lr": 0.001, "wd": 0.0005, "dropout": 0.2, "hidden": 64}')
    a = ap.parse_args()
    grid = [dict(lr=lr, wd=wd, dropout=d, hidden=h) for lr in (5e-3, 1e-3) for wd in (5e-4,)
            for d in (0.2, 0.5) for h in (64,)]
    os.makedirs("results/brats", exist_ok=True)
    log = open("results/brats/train_log.txt", "a")
    if a.stage == "tune":
        for mn in a.models.split(","):
            for cfg in grid:
                f = f"results/brats/tune_{mn}_{cfg['lr']}_{cfg['dropout']}.json"
                if os.path.exists(f):
                    continue
                t = time.time()
                print(f"[tune] {mn} {cfg}", file=log, flush=True)
                _, v = train(mn, cfg, 0, a.root, log=log)
                json.dump({"model": mn, "cfg": cfg, "val": v, "time": time.time() - t}, open(f, "w"))
                print(f"[tune] {mn:14s} {cfg} val_f1={v:.4f} ({time.time()-t:.0f}s)", flush=True)
    else:
        # every method uses the same fixed configuration (no per-method MRI tuning; compute
        # budget), with early stopping on validation patients.
        cfg = json.loads(a.cfg)
        seeds = [int(s) for s in (a.seeds or "100,101,102").split(",")]
        for mn in a.models.split(","):
            for s in seeds:
                f = f"results/brats/final_{mn}_{s}.json"
                if os.path.exists(f):
                    continue
                t = time.time()
                model, v = train(mn, cfg, s, a.root, epochs=12, patience=4, log=log)
                r = {"model": mn, "seed": s, "cfg": cfg, "val": v, "test": test(model, a.root)}
                r["time"] = time.time() - t
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoints/brats_{mn}_{s}.pt")
                json.dump(r, open(f, "w"), indent=1)
                full = r["test"]["FLAIR+T1+T1gd+T2"]["dice"]
                mean_d = np.mean([np.mean(list(v["dice"].values())) for v in r["test"].values()])
                print(f"[final] {mn:16s} s{s} full Dice WT/TC/ET={full['WT']:.3f}/{full['TC']:.3f}/{full['ET']:.3f} "
                      f"mean-over-15 Dice={mean_d:.3f} ({r['time']:.0f}s)", flush=True)

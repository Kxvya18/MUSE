"""MUSE graph benchmarks: Cora / CiteSeer / Amazon-Photo with K=4 feature-split
modalities under per-node random missingness.

Two-stage protocol (identical for every model):
  tune  : grid search scored ONLY on validation accuracy (test never evaluated),
          seeds 0-1; the best config per (model, dataset) is written to configs.json.
  final : the selected config is re-trained with 10 seeds (100-109, disjoint from
          tuning seeds) and evaluated once on test.

Training/selection details:
  * a fresh missing mask each epoch, rate ~ U[0, 0.7] (modality dropout);
  * early stopping on mean val accuracy over fixed masks at rates {0, .3, .6};
  * test: fixed masks at rates 0.0 ... 0.7 + all 2^K - 1 fixed patterns.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

from muse.model import MUSE, muse_loss
from muse.baselines import GCNZero, GATZero, MMGLStyle, TMC, ECD, EGNN, HeMIS, APPNPZero, tmc_loss, ce_loss
from muse.data import load_graph, sample_mask, pattern_mask
from muse.metrics import evaluate

RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MAIN = ["MUSE", "GCN-zero", "GAT-zero", "APPNP-zero", "MMGL-style", "TMC", "ECD", "EGNN", "HeMIS"]
ABLATIONS = ["MUSE-noProp", "MUSE-noAux", "MUSE-selfEvidence", "MUSE-induced", "MUSE-noGate", "MUSE-Dempster", "MUSE-zeroimpute", "MUSE-specVar", "MUSE-noVar"]
GRID = {"lr": [1e-2, 5e-3], "wd": [5e-4, 5e-3], "dropout": [0.3, 0.6], "hidden": [64, 128]}


class ZeroImputeMUSE(MUSE):
    """Ablation: zero-imputation instead of vacuous mass (every modality 'present')."""

    def forward(self, xs, ei, mask):
        xs = [x * mask[:, k:k + 1].float() for k, x in enumerate(xs)]
        return super().forward(xs, ei, torch.ones_like(mask))


def build(name, in_dims, C, hidden=64, dropout=0.6, aux=1.0, act="softplus", ce="logmean", lambda1=0.1,
          prop_steps=10, aux_src="prop", **_):
    kw = dict(hidden=hidden, dropout=dropout)
    mkw = dict(act=act, prop_steps=prop_steps)
    m = {
        "MUSE": lambda: MUSE(in_dims, C, **kw, **mkw),
        "GCN-zero": lambda: GCNZero(in_dims, C, **kw),
        "GAT-zero": lambda: GATZero(in_dims, C, **kw),
        "MMGL-style": lambda: MMGLStyle(in_dims, C, **kw),
        "TMC": lambda: TMC(in_dims, C, **kw),
        "ECD": lambda: ECD(in_dims, C, **kw),
        "EGNN": lambda: EGNN(in_dims, C, **kw),
        "HeMIS": lambda: HeMIS(in_dims, C, **kw),
        "APPNP-zero": lambda: APPNPZero(in_dims, C, **kw),
        "MUSE-MLP": lambda: MUSE(in_dims, C, encoder="mlp", **kw, **mkw),
        "MUSE-noProp": lambda: MUSE(in_dims, C, **kw, **{**mkw, "prop_steps": 0}),
        "MUSE-induced": lambda: MUSE(in_dims, C, mp="induced", evidence="self", **kw, **mkw),  # spec encoder
        "MUSE-selfEvidence": lambda: MUSE(in_dims, C, evidence="self", **kw, **mkw),
        "MUSE-noGate": lambda: MUSE(in_dims, C, use_gate=False, **kw, **mkw),
        "MUSE-Dempster": lambda: MUSE(in_dims, C, combiner="dempster", **kw, **mkw),
        "MUSE-zeroimpute": lambda: ZeroImputeMUSE(in_dims, C, **kw, **mkw),
        "MUSE-specVar": lambda: MUSE(in_dims, C, **kw, **mkw),  # L_var sign as written in the spec
        "MUSE-noVar": lambda: MUSE(in_dims, C, **kw, **mkw),
        "MUSE-noAux": lambda: MUSE(in_dims, C, **kw, **mkw),
    }[name]()
    if name == "TMC":
        loss = tmc_loss
    elif name.startswith("MUSE") or name == "EGNN":
        lkw = {"var_sign": -1.0} if name == "MUSE-specVar" else {}
        if name.startswith("MUSE"):
            lkw["aux"] = 0.0 if name == "MUSE-noAux" else aux
            lkw["ce"], lkw["lambda1"], lkw["aux_src"] = ce, lambda1, aux_src
        if name in ("MUSE-noVar", "EGNN"):
            lkw["lambda2"] = 0.0
        loss = lambda out, y, idx, anneal: muse_loss(out, y, idx, anneal=anneal, **lkw)
    else:
        loss = ce_loss
    return m, loss


def run(dataset, model_name, seed, cfg, test=False, epochs=300, patience=60, K=4, dev="mps", eval_every=None):
    eval_every = eval_every or (5 if dataset == "Photo" else 1)
    torch.manual_seed(seed); np.random.seed(seed)
    data, xs = load_graph(dataset, K, split_seed=seed)
    xs = [x.to(dev) for x in xs]
    y, ei, N, C = data.y.to(dev), data.edge_index.to(dev), data.num_nodes, int(data.y.max()) + 1
    tr, va, te = [m.nonzero(as_tuple=True)[0].to(dev) for m in (data.train_mask, data.val_mask, data.test_mask)]
    model, loss_fn = build(model_name, [x.size(1) for x in xs], C, **cfg)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    g = torch.Generator().manual_seed(1000 + seed)
    val_masks = [sample_mask(N, K, r, g).to(dev) for r in (0.0, 0.3, 0.6)]
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        mask = sample_mask(N, K, float(np.random.uniform(0, 0.7))).to(dev)
        loss, _ = loss_fn(model(xs, ei, mask), y, tr, anneal=min(1.0, ep / 50))
        loss.backward(); opt.step()
        if (ep + 1) % eval_every:
            continue
        model.eval()
        with torch.no_grad():
            v = float(np.mean([evaluate(model(xs, ei, vm), y, va)["acc"] for vm in val_masks]))
        if v > best:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += eval_every
            if bad >= patience:
                break
    res = {"dataset": dataset, "model": model_name, "seed": seed, "cfg": cfg, "val": best, "epochs": ep + 1}
    if not test:
        return res
    model.load_state_dict(best_state); model.eval()
    res.update(rates={}, patterns={})
    gt = torch.Generator().manual_seed(2000 + seed)
    with torch.no_grad():
        for r in RATES:
            res["rates"][str(r)] = evaluate(model(xs, ei, sample_mask(N, K, r, gt).to(dev)), y, te)
        for size in range(1, K + 1):
            for pat in itertools.combinations(range(K), size):
                res["patterns"]["".join(map(str, pat))] = evaluate(model(xs, ei, pattern_mask(N, pat, K).to(dev)), y, te)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_state, f"checkpoints/{dataset}_{model_name}_{seed}.pt")
    return res


def grid():
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]


def cfg_key(c):
    return "_".join(f"{k}{c[k]}" for k in GRID)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "select", "final"])
    ap.add_argument("--datasets", default="Cora,CiteSeer,Photo")
    ap.add_argument("--models", default=",".join(MAIN))
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    datasets, models = a.datasets.split(","), a.models.split(",")

    if a.stage == "tune":
        os.makedirs("results/tune", exist_ok=True)
        seeds = [int(s) for s in (a.seeds or "0,1").split(",")]
        for ds in datasets:
            for mn in models:
                for c in grid():
                    for s in seeds:
                        f = f"results/tune/{ds}_{mn}_{cfg_key(c)}_{s}.json"
                        if os.path.exists(f):
                            continue
                        t = time.time()
                        r = run(ds, mn, s, c)
                        r["time"] = time.time() - t
                        json.dump(r, open(f, "w"))
                        print(f"[tune] {ds:8s} {mn:12s} {cfg_key(c)} s{s} val={r['val']:.4f} ({r['time']:.0f}s)", flush=True)

    elif a.stage == "select":  # best mean-val config per (dataset, model)
        import glob, collections
        agg = collections.defaultdict(list)
        for f in glob.glob("results/tune/*.json"):
            r = json.load(open(f))
            agg[(r["dataset"], r["model"], json.dumps(r["cfg"], sort_keys=True))].append(r["val"])
        best = {}
        for (ds, mn, c), v in agg.items():
            k = f"{ds}|{mn}"
            if k not in best or np.mean(v) > best[k]["val"]:
                best[k] = {"cfg": json.loads(c), "val": float(np.mean(v)), "n": len(v)}
        json.dump(best, open("results/configs.json", "w"), indent=1, sort_keys=True)
        for k, v in sorted(best.items()):
            print(f"{k:28s} val={v['val']:.4f} n={v['n']} {v['cfg']}")

    else:  # final
        os.makedirs("results/final", exist_ok=True)
        configs = json.load(open("results/configs.json"))
        seeds = [int(s) for s in (a.seeds or ",".join(map(str, range(100, 110)))).split(",")]
        for ds in datasets:
            for mn in models:
                # ablations reuse the selected MUSE config (no extra tuning)
                c = configs.get(f"{ds}|{mn}", configs[f"{ds}|MUSE"])["cfg"]
                for s in seeds:
                    f = f"results/final/{ds}_{mn}_{s}.json"
                    if os.path.exists(f):
                        continue
                    t = time.time()
                    r = run(ds, mn, s, c, test=True)
                    r["time"] = time.time() - t
                    json.dump(r, open(f, "w"), indent=1)
                    print(f"[final] {ds:8s} {mn:16s} s{s} acc@0={r['rates']['0.0']['acc']:.4f} "
                          f"acc@.5={r['rates']['0.5']['acc']:.4f} ece@.5={r['rates']['0.5']['ece']:.4f}", flush=True)

"""Empirical order-(in)dependence of fusion rules on real trained evidence.

For each trained MUSE checkpoint (final stage), take the per-modality discounted
Dirichlets on the test nodes with all modalities present and fuse them under 10
random orderings (sequential pairwise combination) with:
  MUSE cumulative sum, Dempster, Yager, PCR5, pairwise averaging.
Reports the max |difference| across orderings of the fused expected probabilities
and of the uncertainty mass, plus the fraction of nodes whose argmax changes.
Also checks Theorem 3 (u non-increasing as modalities are added) on real outputs.
"""
from __future__ import annotations

import glob
import json
import random

import torch

from muse.data import load_graph
from muse.dempster import chain, to_opinion
from run_graph import build

torch.set_default_dtype(torch.float32)
DTYPE = {"float32": torch.float32, "float64": torch.float64}


def muse_chain(disc, order):
    acc = disc[order[0]]
    for k in order[1:]:
        acc = acc + disc[k] - 1
    return to_opinion(acc)


def main():
    out = {}
    for ds in ["Cora", "CiteSeer", "Photo"]:
        ck = sorted(glob.glob(f"checkpoints/{ds}_MUSE_1*.pt"))
        if not ck:
            continue
        cfg = json.load(open("results/configs.json"))[f"{ds}|MUSE"]["cfg"]
        rows = {r: {"prob": [], "u": [], "flip": []} for r in ["MUSE-sum", "dempster", "yager", "pcr5", "average"]}
        mono_viol = []
        for f in ck:
            seed = int(f.split("_")[-1][:-3])
            data, xs = load_graph(ds, 4, split_seed=seed)
            model, _ = build("MUSE", [x.size(1) for x in xs], int(data.y.max()) + 1, **cfg)
            model.load_state_dict(torch.load(f, map_location="cpu")); model.eval()
            mask = torch.ones(data.num_nodes, 4, dtype=torch.bool)
            with torch.no_grad():
                o = model(xs, data.edge_index, mask)
            te = data.test_mask
            for dt in ["float64"]:
                disc = ((o["alphas"] - 1) * o["gammas"] + 1)[:, te].to(DTYPE[dt])
                random.seed(seed)
                orders = [random.sample(range(4), 4) for _ in range(10)]
                for r in rows:
                    res = [muse_chain(disc, od) if r == "MUSE-sum" else chain(disc, od, r) for od in orders]
                    P = torch.stack([b / (b.sum(-1, keepdim=True) + 1e-300) for b, _ in res])  # normalised belief
                    U = torch.stack([u.squeeze(-1) for _, u in res])
                    A = P.argmax(-1)
                    rows[r]["prob"].append((P.max(0).values - P.min(0).values).max().item())
                    rows[r]["u"].append((U.max(0).values - U.min(0).values).max().item())
                    rows[r]["flip"].append((A != A[0]).any(0).float().mean().item())
            # Theorem 3 on real outputs: add modalities one at a time in random order
            al, ga = o["alphas"][:, te].double(), o["gammas"][:, te].double()
            acc = torch.ones_like(al[0])
            u_prev = al.size(-1) / acc.sum(-1)  # vacuous: u = 1
            for k in random.sample(range(4), 4):
                acc = acc + (al[k] - 1) * ga[k]
                u = al.size(-1) / acc.sum(-1)
                mono_viol.append(int((u > u_prev + 1e-12).sum()))
                u_prev = u
        out[ds] = {r: {k: max(v[k]) for k in v} for r, v in rows.items()}
        out[ds]["theorem3_violations"] = sum(mono_viol)
        print(ds, json.dumps(out[ds], indent=1))
    json.dump(out, open("results/order_dependence.json", "w"), indent=1)


if __name__ == "__main__":
    main()

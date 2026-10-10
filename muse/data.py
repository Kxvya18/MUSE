"""Graph benchmarks with K feature-split modalities and per-node missing patterns."""
from __future__ import annotations

import os
import torch
from torch_geometric.datasets import Planetoid, Amazon

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pyg")


def load_graph(name: str, K: int = 4, split_seed: int = 0):
    """Returns data (x, y, edge_index, masks) and xs: list of K modality tensors.

    Feature columns are randomly permuted (fixed seed 0) and split into K equal
    groups; each group is row-normalised independently (a 'modality').
    Planetoid uses the public split; Amazon-Photo uses 20/30 per class train/val.
    """
    if name in ("Cora", "CiteSeer", "PubMed"):
        data = Planetoid(os.path.join(ROOT, "planetoid"), name)[0]
    elif name == "Photo":
        data = Amazon(os.path.join(ROOT, "amazon"), "Photo")[0]
        g = torch.Generator().manual_seed(split_seed)
        C = int(data.y.max()) + 1
        tr, va = [], []
        for c in range(C):
            idx = (data.y == c).nonzero(as_tuple=True)[0]
            idx = idx[torch.randperm(idx.numel(), generator=g)]
            tr.append(idx[:20]); va.append(idx[20:50])
        tr, va = torch.cat(tr), torch.cat(va)
        N = data.num_nodes
        data.train_mask = torch.zeros(N, dtype=torch.bool); data.train_mask[tr] = True
        data.val_mask = torch.zeros(N, dtype=torch.bool); data.val_mask[va] = True
        data.test_mask = ~(data.train_mask | data.val_mask)
    else:
        raise ValueError(name)
    perm = torch.randperm(data.x.size(1), generator=torch.Generator().manual_seed(0))
    xs = []
    for chunk in torch.tensor_split(perm, K):
        x = data.x[:, chunk].float()
        x = x / x.sum(1, keepdim=True).clamp_min(1e-12) if x.min() >= 0 else x
        xs.append(x)
    return data, xs


def sample_mask(N: int, K: int, rate: float, generator=None) -> torch.Tensor:
    """Each (node, modality) missing i.i.d. with prob `rate`; >=1 modality kept per node."""
    m = torch.rand(N, K, generator=generator) >= rate
    empty = ~m.any(1)
    if empty.any():
        keep = torch.randint(0, K, (int(empty.sum()),), generator=generator)
        m[empty.nonzero(as_tuple=True)[0], keep] = True
    return m


def pattern_mask(N: int, pattern: tuple[int, ...], K: int) -> torch.Tensor:
    m = torch.zeros(N, K, dtype=torch.bool)
    m[:, list(pattern)] = True
    return m

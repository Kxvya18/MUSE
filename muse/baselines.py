"""Baselines. All take (xs, edge_index, mask) and return {"logits"} or {"alpha", ...}.

Missing modalities are zero-filled for every baseline (that is what they would see
in deployment); MUSE alone uses vacuous mass instead.

  GCN-zero / GAT-zero : early fusion of zero-imputed concatenated features.
  MMGL-style          : per-modality encoders + learned modality attention fusion
                        (simplified re-implementation in the spirit of MMGL,
                        Zheng et al. MedIA 2022; no adaptive graph learning).
  TMC                 : Trusted Multi-view Classification (Han et al. TPAMI 2022):
                        per-modality evidential heads + Dempster's rule, per-view
                        and fused EDL losses.
  ECD                 : evidential fusion with contextual discounting in the style
                        of Huang & Denoeux (learned global per-modality/per-class
                        discount) + Dempster's rule.
  EGNN                : evidential GNN on concatenated input (MUSE w/o combiner).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .model import GATv2Encoder, EvidentialHead, kl_dirichlet_uniform
from .dempster import chain, to_dirichlet


def zero_fill(xs, mask):
    return [x * mask[:, k:k + 1].float() for k, x in enumerate(xs)]


class GCNZero(nn.Module):
    def __init__(self, in_dims, C, hidden=64, dropout=0.5, **_):
        super().__init__()
        self.c1, self.c2, self.p = GCNConv(sum(in_dims), hidden), GCNConv(hidden, C), dropout

    def forward(self, xs, ei, mask):
        x = torch.cat(zero_fill(xs, mask), -1)
        x = F.dropout(x, self.p, self.training)
        x = F.dropout(F.relu(self.c1(x, ei)), self.p, self.training)
        return {"logits": self.c2(x, ei)}


class GATZero(nn.Module):
    def __init__(self, in_dims, C, hidden=64, heads=4, dropout=0.6, **_):
        super().__init__()
        self.enc, self.out = GATv2Encoder(sum(in_dims), hidden, heads, dropout), nn.Linear(hidden, C)

    def forward(self, xs, ei, mask):
        return {"logits": self.out(F.elu(self.enc(torch.cat(zero_fill(xs, mask), -1), ei)))}


class MMGLStyle(nn.Module):
    def __init__(self, in_dims, C, hidden=64, heads=4, dropout=0.6, **_):
        super().__init__()
        self.encs = nn.ModuleList([GATv2Encoder(d, hidden, heads, dropout) for d in in_dims])
        self.att = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
        self.out = nn.Linear(hidden, C)

    def forward(self, xs, ei, mask):
        H = torch.stack([F.elu(e(x, ei)) for e, x in zip(self.encs, zero_fill(xs, mask))])  # K,N,F
        w = torch.softmax(self.att(H), 0)
        return {"logits": self.out((w * H).sum(0)), "weights": w.squeeze(-1).t()}


class TMC(nn.Module):
    def __init__(self, in_dims, C, hidden=64, heads=4, dropout=0.6, **_):
        super().__init__()
        self.encs = nn.ModuleList([GATv2Encoder(d, hidden, heads, dropout) for d in in_dims])
        self.heads = nn.ModuleList([EvidentialHead(hidden, C) for _ in in_dims])

    def forward(self, xs, ei, mask):
        alphas = torch.stack([h(e(x, ei)) for e, h, x in zip(self.encs, self.heads, zero_fill(xs, mask))])
        b, u = chain(alphas, list(range(alphas.size(0))), "dempster")
        return {"alpha": to_dirichlet(b, u), "alphas": alphas}


class ECD(nn.Module):
    """Contextual discounting: m_k discounted by learned beta_k in (0,1)^C, then Dempster."""

    def __init__(self, in_dims, C, hidden=64, heads=4, dropout=0.6, **_):
        super().__init__()
        self.encs = nn.ModuleList([GATv2Encoder(d, hidden, heads, dropout) for d in in_dims])
        self.heads = nn.ModuleList([EvidentialHead(hidden, C) for _ in in_dims])
        # same per-modality LayerNorm as MUSE: without it ECD stays at chance on Amazon-Photo
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in in_dims])
        self.beta = nn.Parameter(torch.full((len(in_dims), C), 2.0))

    def forward(self, xs, ei, mask):
        alphas = torch.stack([h(n(e(x, ei))) for e, h, n, x in zip(self.encs, self.heads, self.norms, zero_fill(xs, mask))])
        beta = torch.sigmoid(self.beta).unsqueeze(1)  # K,1,C
        disc = (alphas - 1) * beta + 1  # discounting moves belief to ignorance
        b, u = chain(disc, list(range(disc.size(0))), "dempster")
        return {"alpha": to_dirichlet(b, u), "alphas": disc, "gammas": beta.expand_as(alphas)}


class EGNN(nn.Module):
    def __init__(self, in_dims, C, hidden=64, heads=4, dropout=0.6, **_):
        super().__init__()
        self.enc, self.head = GATv2Encoder(sum(in_dims), hidden, heads, dropout), EvidentialHead(hidden, C)
        self.norm = nn.LayerNorm(hidden)  # same normalisation as MUSE/ECD; without it EGNN collapses on Photo

    def forward(self, xs, ei, mask):
        return {"alpha": self.head(self.norm(self.enc(torch.cat(zero_fill(xs, mask), -1), ei)))}


# ------------------------------------------------------------------ losses
def edl_ce(alpha, y, C):
    """Bayes-risk cross-entropy (Sensoy et al. 2018, as used by TMC)."""
    yo = F.one_hot(y, C).float()
    S = alpha.sum(-1, keepdim=True)
    return (yo * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)


def tmc_loss(out, y, idx, anneal, lam=0.1, **_):
    C = out["alpha"].size(-1)
    yo = F.one_hot(y[idx], C).float()
    terms = [out["alpha"][idx]] + [a[idx] for a in out["alphas"]]
    loss = 0
    for a in terms:
        at = yo + (1 - yo) * a
        loss = loss + (edl_ce(a, y[idx], C) + anneal * lam * kl_dirichlet_uniform(at)).mean()
    return loss, {}


def ce_loss(out, y, idx, **_):
    if "logits" in out:
        return F.cross_entropy(out["logits"][idx], y[idx]), {}
    a = out["alpha"][idx]
    return F.nll_loss(torch.log(a / a.sum(-1, keepdim=True)), y[idx]), {}


class HeMIS(nn.Module):
    """HeMIS-style fusion (Havaei et al., MICCAI 2016): per-modality encoders, fused by
    the first and second moments (mean, variance) of the features over the modalities
    available at each node, then a classifier. No imputation. Uses exactly MUSE's
    encoders and availability masks, so it differs from MUSE only in the fusion rule."""

    def __init__(self, in_dims, C, hidden=64, heads=4, dropout=0.6, evidence="neighbourhood", **_):
        super().__init__()
        from .model import MUSE
        self.enc = MUSE(in_dims, C, hidden=hidden, heads=heads, dropout=dropout, evidence=evidence)
        self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, C))

    def forward(self, xs, ei, mask):
        H, avail = self.enc.encode(xs, ei, mask)
        m = avail.t().unsqueeze(-1).float()
        n = m.sum(0).clamp_min(1.0)
        mu = (H * m).sum(0) / n
        var = (((H - mu) ** 2) * m).sum(0) / n
        return {"logits": self.out(torch.cat([mu, var], -1))}


class APPNPZero(nn.Module):
    """APPNP (Gasteiger et al., ICLR 2019) on zero-imputed concatenated features:
    MLP predictions propagated with the same PPR operator/steps/teleport as MUSE."""

    def __init__(self, in_dims, C, hidden=64, dropout=0.5, steps=10, teleport=0.1, **_):
        super().__init__()
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(sum(in_dims), hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, C))
        self.steps, self.teleport = steps, teleport

    def forward(self, xs, ei, mask):
        from .model import propagate_evidence
        z = self.mlp(torch.cat(zero_fill(xs, mask), -1))
        return {"logits": propagate_evidence(z, ei, self.steps, self.teleport)}

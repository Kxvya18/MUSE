"""Training losses. Every model gets soft Dice on its fused probabilities plus its own
natural likelihood term, so the comparison is about fusion, not about the loss recipe."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from muse.model import kl_dirichlet_uniform


def soft_dice(prob, y, eps=1e-5):
    """Mean soft Dice loss over the tumour classes 1..C-1."""
    yo = F.one_hot(y, prob.size(1)).movedim(-1, 1).float()
    dims = (0, 2, 3, 4)
    inter = (prob * yo).sum(dims)
    den = prob.sum(dims) + yo.sum(dims)
    return 1 - ((2 * inter + eps) / (den + eps))[1:].mean()


def evidential_terms(alpha, y, anneal, lam_kl=0.1):
    """Spec loss per voxel: -log(alpha_y / S) + anneal * lam * KL[Dir(alpha~) || Dir(1)]."""
    yo = F.one_hot(y, alpha.size(1)).movedim(-1, 1).float()
    S = alpha.sum(1, keepdim=True)
    ce = -(yo * torch.log(alpha / S)).sum(1).mean()
    a_t = (yo + (1 - yo) * alpha).movedim(1, -1)
    return ce + anneal * lam_kl * kl_dirichlet_uniform(a_t).mean()


def muse_loss(out, y, anneal, aux=1.0, lam_var=0.05):
    alpha = out["alpha"]
    loss = evidential_terms(alpha, y, anneal) + soft_dice(alpha / alpha.sum(1, keepdim=True), y)
    # deep supervision (MUSE2-ctx-ds): fused evidence at 1/4 and 1/2 resolution
    for a_s, w in zip(out.get("alpha_ds", []), (0.25, 0.5)):
        step = y.shape[-1] // a_s.shape[-1]
        y_s = y[:, ::step, ::step, ::step]
        loss = loss + w * (evidential_terms(a_s, y_s, anneal) + soft_dice(a_s / a_s.sum(1, keepdim=True), y_s))
    # per-sequence evidence must be valid on its own (only where the sequence was acquired)
    mask = out["mask"]
    for k in range(4):
        idx = mask[:, k].nonzero(as_tuple=True)[0]
        if idx.numel():
            loss = loss + aux / 4 * evidential_terms(out["alphas"][k][idx], y[idx], anneal)
    # balance regulariser: penalise the spread of reliabilities across present sequences
    if lam_var:
        g, m = out["gammas"], mask.t().float()[:, :, None, None, None, None]  # K, B, C|1, ...
        n = m.sum(0)                       # B, 1, 1, 1, 1
        several = n.flatten() >= 2         # samples with at least two sequences
        if several.any():
            mu = (g * m).sum(0) / n.clamp_min(1)
            var = (((g - mu) ** 2) * m).sum(0) / n.clamp_min(1)
            loss = loss + lam_var * var[several].mean()
    return loss


def tmc_loss(out, y, anneal):
    yo = F.one_hot(y, out["alpha"].size(1)).movedim(-1, 1).float()

    def edl(a, yy, yoo):
        S = a.sum(1, keepdim=True)
        ce = (yoo * (torch.digamma(S) - torch.digamma(a))).sum(1).mean()
        return ce + anneal * 0.1 * kl_dirichlet_uniform((yoo + (1 - yoo) * a).movedim(1, -1)).mean()

    a = out["alpha"]
    loss = edl(a, y, yo) + soft_dice(a / a.sum(1, keepdim=True), y)
    for k in range(4):
        idx = out["mask"][:, k].nonzero(as_tuple=True)[0]
        if idx.numel():
            loss = loss + edl(out["alphas"][k][idx], y[idx], yo[idx]) / 4
    return loss


def softmax_loss(out, y, anneal):
    if "prob" in out:  # LateMean: NLL of the averaged probability + per-sequence CE
        p = out["prob"]
        loss = F.nll_loss(torch.log(p.clamp_min(1e-8)), y) + soft_dice(p, y)
        for k in range(4):
            idx = out["mask"][:, k].nonzero(as_tuple=True)[0]
            if idx.numel():
                loss = loss + F.cross_entropy(out["logits_k"][k][idx], y[idx]) / 4
        return loss
    logits = out["logits"].float()
    loss = F.cross_entropy(logits, y) + soft_dice(torch.softmax(logits, 1), y)
    if "aux_logits" in out:  # RFNet-style per-sequence auxiliary heads
        for k in range(4):
            idx = out["mask"][:, k].nonzero(as_tuple=True)[0]
            if idx.numel():
                loss = loss + F.cross_entropy(out["aux_logits"][k][idx], y[idx]) / 4
    return loss


def loss_for(name):
    if name.startswith("MUSE"):
        return muse_loss
    if name == "TMC":
        return tmc_loss
    return softmax_loss

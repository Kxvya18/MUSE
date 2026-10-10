from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def outputs_to_prob_unc(out):
    """Common interface: expected class probs and an uncertainty score per node."""
    if "alpha" in out:
        a = out["alpha"]
        S = a.sum(-1)
        return a / S.unsqueeze(-1), a.size(-1) / S  # u = C/S
    p = torch.softmax(out["logits"], -1)
    return p, -(p * p.clamp_min(1e-12).log()).sum(-1) / np.log(p.size(-1))  # normalised entropy


def ece(prob: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    conf, pred = prob.max(-1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.float().mean().item() * abs(acc[s].mean().item() - conf[s].mean().item())
    return e


def reliability(prob, y, bins=15):
    conf, pred = prob.max(-1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        rows.append([int(s.sum()), acc[s].mean().item() if s.any() else 0.0, conf[s].mean().item() if s.any() else 0.0])
    return rows  # [count, accuracy, mean confidence] per bin


def evaluate(out, y, idx, bins=False):
    prob, unc = outputs_to_prob_unc(out)
    prob, unc, yy = prob[idx].detach().cpu(), unc[idx].detach().cpu(), y[idx].cpu()
    pred = prob.argmax(-1)
    correct = (pred == yy)
    nll = -torch.log(prob[torch.arange(len(yy)), yy].clamp_min(1e-12)).mean().item()
    try:  # uncertainty should be higher on errors
        auroc = roc_auc_score((~correct).numpy(), unc.numpy()) if 0 < correct.sum() < len(yy) else float("nan")
    except ValueError:
        auroc = float("nan")
    res = {
        "acc": correct.float().mean().item(),
        "f1": f1_score(yy.numpy(), pred.numpy(), average="macro"),
        "ece": ece(prob, yy),
        "nll": nll,
        "auroc_err": auroc,
        "unc": unc.mean().item(),
    }
    if bins:
        res["reliability"] = reliability(prob, yy)
    return res

"""Classical combination rules on subjective-logic opinions (singletons + Theta).

A Dirichlet alpha over C classes maps to the mass function
    m({c}) = b_c = (alpha_c - 1)/S,   m(Theta) = u = C/S,
the representation used by TMC (Han et al.) and evidential MRI fusion
(Huang et al.). These rules are used (i) as baselines/ablations and (ii) to
measure order-dependence empirically when chained over N > 2 sources.
"""
from __future__ import annotations

import torch

EPS = 1e-12


def to_opinion(alpha):
    S = alpha.sum(-1, keepdim=True)
    return (alpha - 1) / S, alpha.size(-1) / S


def to_dirichlet(b, u):
    S = b.size(-1) / u.clamp_min(EPS)
    return b * S + 1


def dempster_pair(b1, u1, b2, u2):
    """Dempster's rule (normalised conjunctive) restricted to singleton+Theta masses."""
    conflict = b1.sum(-1, keepdim=True) * b2.sum(-1, keepdim=True) - (b1 * b2).sum(-1, keepdim=True)
    z = (1 - conflict).clamp_min(EPS)
    b = (b1 * b2 + b1 * u2 + b2 * u1) / z
    u = (u1 * u2) / z
    return b, u


def yager_pair(b1, u1, b2, u2):
    """Yager's rule: conflict mass assigned to Theta instead of normalising."""
    conflict = b1.sum(-1, keepdim=True) * b2.sum(-1, keepdim=True) - (b1 * b2).sum(-1, keepdim=True)
    return b1 * b2 + b1 * u2 + b2 * u1, u1 * u2 + conflict


def pcr5_pair(b1, u1, b2, u2):
    """PCR5 (Smarandache-Dezert) on singleton+Theta masses: partial conflict is
    redistributed proportionally to the two singletons involved."""
    b = b1 * b2 + b1 * u2 + b2 * u1
    # conflict between {i} from m1 and {j} from m2, i != j
    B1, B2 = b1.unsqueeze(-1), b2.unsqueeze(-2)  # [...,C,1] (m1's {i}), [...,1,C] (m2's {j})
    off = 1 - torch.eye(b1.size(-1), device=b1.device, dtype=b1.dtype)
    den = (B1 + B2).clamp_min(EPS)
    to_i = (B1 ** 2 * B2 / den * off).sum(-1)  # share of conflict {i}x{j} returned to i
    to_j = (B1 * B2 ** 2 / den * off).sum(-2)  # share returned to j
    return b + to_i + to_j, u1 * u2


def average_pair(b1, u1, b2, u2):
    """Pairwise opinion averaging (averaging belief fusion applied sequentially)."""
    return (b1 + b2) / 2, (u1 + u2) / 2


RULES = {"dempster": dempster_pair, "yager": yager_pair, "pcr5": pcr5_pair, "average": average_pair}


def chain(alphas, order, rule="dempster"):
    """Sequentially combine Dirichlets alphas[k] (k in order) with a classical rule."""
    f = RULES[rule]
    b, u = to_opinion(alphas[order[0]])
    for k in order[1:]:
        b2, u2 = to_opinion(alphas[k])
        b, u = f(b, u, b2, u2)
    return b, u


def dempster_fuse_dirichlet(alphas):
    """Fuse [K,N,C] Dirichlets with Dempster's rule, returning a Dirichlet [N,C]."""
    b, u = chain(alphas, list(range(alphas.size(0))), "dempster")
    return to_dirichlet(b, u)

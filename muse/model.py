"""MUSE: Modality-Universal Selective Evidence Fusion.

Implements the five components of the MUSE specification exactly:
  1. Per-modality GATv2 encoders (2 layers, H=4 heads, dropout 0.6), run only on
     nodes that observe the modality (induced subgraph -> no imputation leaks in).
  2. Evidential head: alpha = softplus(W2 ReLU(W1 h + b1) + b2) + 1.
  3. Vacuous mass for missing modalities: alpha = 1 (no encoder pass).
  4. Contextual reliability gate: gamma = sigmoid(W [h || c] + b), with
     c_i = mean_{j in N(i)} mean_{k in O_j} h_j^(k); gamma = 0 for missing.
  5. N-modal combiner: alpha_fused = sum_{k in O} (alpha^(k) - 1) * gamma^(k) + 1.

Loss: L = L_CE + lambda1 * L_KL + lambda2 * L_var (spec Sec. 7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import subgraph, add_self_loops, scatter


class GATv2Encoder(nn.Module):
    """Spec Component 1: GATv2Conv(d_k, F/H, heads=H) -> GATv2Conv(F, F, heads=1)."""

    def __init__(self, in_dim: int, hidden: int = 64, heads: int = 4, dropout: float = 0.6,
                 in_dropout: float | None = None):
        super().__init__()
        self.dropout = dropout
        self.in_dropout = dropout if in_dropout is None else in_dropout
        self.conv1 = GATv2Conv(in_dim, hidden // heads, heads=heads, dropout=dropout, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden, hidden, heads=1, dropout=dropout, add_self_loops=False)

    def forward(self, x, edge_index, edge_index2=None):
        """edge_index (layer 1) / edge_index2 (layer 2). If edge_index2 is None, both
        layers use edge_index plus self-loops (standard GATv2)."""
        if edge_index2 is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            edge_index2 = edge_index
        x = F.dropout(x, self.in_dropout, self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        return self.conv2(x, edge_index2)


class MLPEncoder(nn.Module):
    """Per-modality node-wise encoder (no message passing); graph context enters
    through evidence propagation after fusion."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.lin1, self.lin2, self.p = nn.Linear(in_dim, hidden), nn.Linear(hidden, hidden), dropout

    def forward(self, x, *_):
        x = F.dropout(x, self.p, self.training)
        x = F.dropout(F.elu(self.lin1(x)), self.p, self.training)
        return self.lin2(x)


class EvidentialHead(nn.Module):
    """Spec Component 2: two-layer MLP -> Dirichlet concentration (alpha > 1).

    act="softplus" (spec) or "exp" (clamped exp evidence, which avoids the zero-gradient
    'dead evidence' regions of softplus/ReLU heads; Pandey & Yu, ICML 2023).
    """

    def __init__(self, hidden: int, num_classes: int, act: str = "softplus"):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.act = act

    def forward(self, h):
        z = self.fc2(F.relu(self.fc1(h)))
        e = torch.exp(z.clamp(-10, 10)) if self.act == "exp" else F.softplus(z)
        return e + 1.0


class ReliabilityGate(nn.Module):
    """Spec Component 3: gamma = sigmoid(W_gamma [h || c || o] + b_gamma) in (0,1)^C.

    o in {0,1} flags whether the node observed the modality itself (1) or only
    receives it from its neighbourhood (0), so second-hand evidence can be discounted.
    """

    def __init__(self, hidden: int, num_classes: int):
        super().__init__()
        self.lin = nn.Linear(2 * hidden + 1, num_classes)

    def forward(self, h, c, o):
        return torch.sigmoid(self.lin(torch.cat([h, c, o], dim=-1)))


def n_modal_combine(alphas: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    """Spec Component 4: alpha_fused = sum_k (alpha_k - 1) * gamma_k + 1.

    alphas, gammas: [K, N, C]. Missing modalities carry alpha=1 (and gamma=0), so
    their excess evidence is exactly zero (vacuous identity).
    """
    return ((alphas - 1.0) * gammas).sum(0) + 1.0


def propagate_evidence(E: torch.Tensor, edge_index: torch.Tensor, steps: int = 10, teleport: float = 0.1):
    """Personalised-PageRank propagation of excess evidence E >= 0 ([..., N, C]) with the
    random-walk operator D^-1 (A + I) (mass-preserving, so S stays on the same scale):
        E^{t+1} = (1 - teleport) P E^t + teleport E^0.
    Linear with non-negative weights, hence (i) it commutes with the N-modal sum
    (associativity and per-modality attribution are preserved), (ii) zero evidence
    stays zero (vacuous identity), (iii) more evidence never lowers S (monotonicity)."""
    N = E.size(-2)
    ei, _ = add_self_loops(edge_index, num_nodes=N)
    src, dst = ei
    deg = scatter(torch.ones_like(dst, dtype=E.dtype), dst, dim=0, dim_size=N, reduce="sum")
    w = (1.0 / deg)[dst]
    H = E
    for _ in range(steps):
        msg = H.index_select(-2, src) * w.unsqueeze(-1)
        agg = torch.zeros_like(H).index_add_(-2, dst, msg)
        H = (1 - teleport) * agg + teleport * E
    return H


def combine_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Binary form of the combiner on already-fused Dirichlets (Theorem 1)."""
    return (a - 1.0) + (b - 1.0) + 1.0


class MUSE(nn.Module):
    def __init__(self, in_dims: list[int], num_classes: int, hidden: int = 64, heads: int = 4,
                 dropout: float = 0.6, in_dropout: float | None = None, use_gate: bool = True,
                 combiner: str = "sum", mp: str = "observed", evidence: str = "neighbourhood",
                 act: str = "softplus", prop_steps: int = 10, teleport: float = 0.1, encoder: str = "gatv2",
                 norm: bool = True):
        super().__init__()
        self.K, self.C, self.F = len(in_dims), num_classes, hidden
        self.use_gate, self.combiner, self.mp, self.evidence = use_gate, combiner, mp, evidence
        self.prop_steps, self.teleport = prop_steps, teleport
        self.encoder = encoder
        if encoder == "mlp":
            self.encoders = nn.ModuleList([MLPEncoder(d, hidden, dropout) for d in in_dims])
        else:
            self.encoders = nn.ModuleList([GATv2Encoder(d, hidden, heads, dropout, in_dropout) for d in in_dims])
        self.heads = nn.ModuleList([EvidentialHead(hidden, num_classes, act) for _ in in_dims])
        # per-modality LayerNorm on encoder outputs: makes each source's evidence invariant
        # to its input scale (without it, low-magnitude modalities give near-constant
        # embeddings and vanishing evidence gradients, e.g. dense bag-of-words on Photo)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) if norm else nn.Identity() for _ in in_dims])
        self.gates = nn.ModuleList([ReliabilityGate(hidden, num_classes) for _ in in_dims])

    def encode(self, xs, edge_index, mask):
        """Per-modality encoding without imputation.

        Returns H [K,N,F] and avail [N,K]: where modality k yields evidence at node i.
        evidence="self" (spec): avail = mask, i.e. only nodes observing k.
        evidence="neighbourhood" (default, graphs): k yields evidence at i whenever an
          observed k-feature lies in i's receptive field (i itself or its <=2-hop
          in-neighbourhood). i's own missing features are never read; if nothing in
          the receptive field observes k, k is vacuous at i (alpha = 1).

        mp="induced": encoder k runs on the induced subgraph of nodes observing k.
        mp="observed" (default): observation-aware message passing on the full graph.
          Layer 1 only carries messages *from* nodes observing k (plus their self-loops),
          so a raw feature of a missing modality is never read; layer 2 carries messages
          from every node whose layer-1 state is defined (it observes k or has an
          observing in-neighbour). Nodes missing k can thus relay their neighbours'
          observed k-evidence, which keeps the graph connected under heavy missingness.
        """
        N = mask.size(0)
        H = xs[0].new_zeros(self.K, N, self.F)
        avail = mask.clone()
        src, dst = edge_index
        loop = torch.arange(N, device=edge_index.device)
        for k in range(self.K):
            obs = mask[:, k]
            idx = obs.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            if self.encoder == "mlp":  # node-wise: evidence only where k is observed
                H[k, idx] = self.encoders[k](xs[k][idx])
                continue
            if self.mp == "induced":
                ei, _ = subgraph(idx, edge_index, relabel_nodes=True, num_nodes=N)
                H[k, idx] = self.encoders[k](xs[k][idx], ei)
                continue
            keep1 = obs[src]
            ei1 = torch.cat([edge_index[:, keep1], torch.stack([idx, idx])], 1)
            defined = obs.clone()
            defined[dst[keep1]] = True
            keep2 = defined[src]
            didx = loop[defined]
            ei2 = torch.cat([edge_index[:, keep2], torch.stack([didx, didx])], 1)
            x = torch.where(obs.unsqueeze(-1), xs[k], torch.zeros_like(xs[k]))  # never sent anyway
            hk = self.encoders[k](x, ei1, ei2)
            if self.evidence == "neighbourhood":
                reach = defined.clone()
                reach[dst[keep2]] = True
                avail[:, k] = reach
                H[k, reach] = hk[reach]
            else:
                H[k, idx] = hk[idx]
        return H, avail

    def forward(self, xs, edge_index, mask):
        """xs: list of K tensors [N, d_k]; mask: [N, K] bool (True = observed).

        Returns dict with alpha_fused [N,C], per-modality alphas/gammas [K,N,C].
        """
        N = mask.size(0)
        H, avail = self.encode(xs, edge_index, mask)
        H = torch.stack([self.norms[k](H[k]) for k in range(self.K)])
        m = avail.t().unsqueeze(-1).float()  # [K, N, 1]: 1 where k yields evidence
        o = mask.t().unsqueeze(-1).float()   # [K, N, 1]: 1 where node observed k itself
        # Component 2/3: evidential head on present modalities, vacuous alpha=1 otherwise.
        alphas = torch.stack([self.heads[k](H[k]) for k in range(self.K)])
        alphas = torch.where(m.bool(), alphas, torch.ones_like(alphas))
        # Context c_i: mean over neighbours (incl. self) of mean over present modalities.
        own = (H * m).sum(0) / m.sum(0).clamp_min(1.0)
        ei, _ = add_self_loops(edge_index, num_nodes=N)
        c = scatter(own[ei[0]], ei[1], dim=0, dim_size=N, reduce="mean")
        if self.use_gate:
            gammas = torch.stack([self.gates[k](H[k], c, o[k]) for k in range(self.K)])
        else:
            gammas = torch.ones_like(alphas)
        gammas = gammas * m  # gamma = 0 for missing modalities
        if self.combiner == "sum":
            alpha_f = n_modal_combine(alphas, gammas)
        elif self.combiner == "dempster":
            from .dempster import dempster_fuse_dirichlet
            alpha_f = dempster_fuse_dirichlet((alphas - 1) * gammas + 1)
        else:
            raise ValueError(self.combiner)
        out_extra = {}
        if self.prop_steps > 0:
            if self.combiner == "sum":
                # propagate each source's excess evidence (raw and discounted); by linearity
                # sum_k P(excess_k) == P(sum_k excess_k), so the fused result is unchanged
                # while each modality's graph-propagated evidence stays available for the
                # per-modality loss and for attribution.
                raw = propagate_evidence(alphas - 1, edge_index, self.prop_steps, self.teleport)
                disc = propagate_evidence((alphas - 1) * gammas, edge_index, self.prop_steps, self.teleport)
                alpha_f = disc.sum(0) + 1
                out_extra = {"alphas_prop": raw + 1, "contrib": disc}
            else:
                alpha_f = propagate_evidence(alpha_f - 1, edge_index, self.prop_steps, self.teleport) + 1
        return {"alpha": alpha_f, "alphas": alphas, "gammas": gammas, "mask": avail, "observed": mask, **out_extra}


# ---------------------------------------------------------------- outputs & loss
def dirichlet_outputs(alpha: torch.Tensor):
    """Spec Sec. 7: S, belief b, uncertainty u = C/S, prediction argmax b, prob alpha/S."""
    S = alpha.sum(-1, keepdim=True)
    b = (alpha - 1) / S
    u = alpha.size(-1) / S.squeeze(-1)
    return {"S": S.squeeze(-1), "belief": b, "u": u, "pred": b.argmax(-1), "prob": alpha / S}


def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL[Dir(alpha) || Dir(1)] per row."""
    C = alpha.size(-1)
    S = alpha.sum(-1)
    t1 = torch.lgamma(S) - torch.lgamma(torch.tensor(float(C), device=alpha.device)) - torch.lgamma(alpha).sum(-1)
    t2 = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S).unsqueeze(-1))).sum(-1)
    return t1 + t2


def _ce(alpha, yo, kind):
    """logmean (spec): -log E[p_y] = -log(alpha_y/S); digamma: Bayes risk of CE
    E_Dir[-log p_y] = psi(S) - psi(alpha_y) (Sensoy et al. 2018)."""
    S = alpha.sum(-1, keepdim=True)
    if kind == "digamma":
        return (yo * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)
    return -(yo * torch.log(alpha / S)).sum(-1)


def muse_loss(out, y, idx, lambda1: float = 0.1, lambda2: float = 0.05, anneal: float = 1.0,
              var_sign: float = 1.0, aux: float = 0.0, ce: str = "logmean", aux_src: str = "prop"):
    """L = L_CE + anneal * lambda1 L_KL + lambda2 L_var.

    var_sign=+1 (default) penalises the spread of gamma across present modalities,
    i.e. a genuine *balance* regulariser. The written spec uses -Var, which when
    minimised maximises the spread (the opposite of balance); var_sign=-1
    reproduces it for ablation. `anneal` in [0,1] is the standard EDL KL warm-up.
    """
    alpha = out["alpha"][idx]
    yo = F.one_hot(y[idx], alpha.size(-1)).float()
    l_ce = _ce(alpha, yo, ce).mean()
    alpha_t = yo + (1 - yo) * alpha
    l_kl = kl_dirichlet_uniform(alpha_t).mean()
    # Var_{k in O_i}[gamma_i^(k)], averaged over classes and nodes with |O_i| >= 2.
    l_var = alpha.new_zeros(())
    if lambda2 != 0 and "gammas" in out:  # models without a reliability gate (e.g. EGNN) skip it
        g, m = out["gammas"][:, idx], out["mask"][idx].t().unsqueeze(-1).float()
        n = m.sum(0)
        valid = (n.squeeze(-1) >= 2)
        if valid.any():
            mu = (g * m).sum(0) / n.clamp_min(1)
            var = (((g - mu) ** 2) * m).sum(0) / n.clamp_min(1)
            l_var = var_sign * var[valid].mean()
    loss = l_ce + anneal * lambda1 * l_kl + lambda2 * l_var
    if aux > 0:
        # per-modality evidential terms on the (undiscounted) source evidence, only
        # where the modality is observed: each source must be informative on its own.
        use_prop = aux_src == "prop" and "alphas_prop" in out
        src = out["alphas_prop"] if use_prop else out["alphas"]
        if use_prop:  # propagated source evidence exists wherever it reached
            mk = (src[:, idx] - 1).sum(-1) > 1e-6
        else:
            mk = out["mask"][idx].t()  # K, n
        l_aux = alpha.new_zeros(())
        for k in range(mk.size(0)):
            sel = mk[k]
            if sel.any():
                ak, yk = src[k][idx][sel], yo[sel]
                l_aux = l_aux + _ce(ak, yk, ce).mean()
                l_aux = l_aux + anneal * lambda1 * kl_dirichlet_uniform(yk + (1 - yk) * ak).mean()
        loss = loss + aux * l_aux / mk.size(0)
    return loss, {"ce": l_ce.item(), "kl": l_kl.item(), "var": l_var.item()}

import itertools
import random

import torch

from muse.model import MUSE, n_modal_combine, combine_pair, dirichlet_outputs
from muse.dempster import chain
from muse.data import sample_mask


def rand_alpha(K=4, N=50, C=5):
    return torch.rand(K, N, C, dtype=torch.float64) * 20 + 1, torch.rand(K, N, C, dtype=torch.float64)


def test_associativity_any_order_and_grouping():
    a, g = rand_alpha()
    ref = n_modal_combine(a, g)
    disc = (a - 1) * g + 1
    for _ in range(10):
        order = random.sample(range(4), 4)
        acc = disc[order[0]]
        for k in order[1:]:
            acc = combine_pair(acc, disc[k])
        assert torch.allclose(acc, ref, atol=1e-10)
    # grouping A | B
    for r in range(1, 4):
        for A in itertools.combinations(range(4), r):
            B = [k for k in range(4) if k not in A]
            fa, fb = n_modal_combine(a[list(A)], g[list(A)]), n_modal_combine(a[B], g[B])
            assert torch.allclose(combine_pair(fa, fb), ref, atol=1e-10)


def test_vacuous_identity():
    a, g = rand_alpha()
    ref = n_modal_combine(a, g)
    vac = torch.ones_like(a[:1])
    assert torch.equal(n_modal_combine(torch.cat([a, vac]), torch.cat([g, torch.rand_like(vac)])), ref)


def test_monotone_uncertainty():
    a, g = rand_alpha()
    for k in range(1, 5):
        u_prev = dirichlet_outputs(n_modal_combine(a[:k - 1], g[:k - 1]) if k > 1 else torch.ones_like(a[0]))["u"]
        u = dirichlet_outputs(n_modal_combine(a[:k], g[:k]))["u"]
        assert (u <= u_prev + 1e-12).all()


def test_belief_plus_uncertainty_is_one():
    a, _ = rand_alpha()
    o = dirichlet_outputs(a[0])
    assert torch.allclose(o["belief"].sum(-1) + o["u"], torch.ones_like(o["u"]))


def test_dempster_is_associative_too():
    """Dempster's rule is associative; the order-independence of MUSE is NOT a
    differentiator against Dempster (only against Yager/PCR5/averaging)."""
    a, _ = rand_alpha()
    b0, u0 = chain(a, [0, 1, 2, 3])
    b1, u1 = chain(a, [3, 1, 0, 2])
    assert torch.allclose(b0, b1, atol=1e-10) and torch.allclose(u0, u1, atol=1e-10)


def test_missing_modality_does_not_touch_encoder():
    torch.manual_seed(0)
    N, C = 30, 3
    xs = [torch.rand(N, 6) for _ in range(3)]
    ei = torch.randint(0, N, (2, 80))
    m = MUSE([6, 6, 6], C).eval()
    mask = sample_mask(N, 3, 0.5)
    out = m(xs, ei, mask)
    xs2 = [x.clone() for x in xs]
    xs2[1][~mask[:, 1]] = 1e6  # garbage in missing slots must not change anything
    out2 = m(xs2, ei, mask)
    assert torch.allclose(out["alpha"], out2["alpha"])
    av = out["mask"].t()
    assert (out["alphas"][~av] == 1).all() and (out["gammas"][~av] == 0).all()
    assert (av | ~mask.t()).all() and av[mask.t()].all()  # observed => available


def test_spec_self_evidence_mode():
    torch.manual_seed(0)
    N = 30
    xs = [torch.rand(N, 6) for _ in range(3)]
    ei = torch.randint(0, N, (2, 80))
    m = MUSE([6, 6, 6], 3, evidence="self").eval()
    mask = sample_mask(N, 3, 0.5)
    out = m(xs, ei, mask)
    assert (out["alphas"][~mask.t()] == 1).all()


def test_isolated_unobserved_node_is_vacuous():
    xs = [torch.rand(3, 4) for _ in range(2)]
    ei = torch.tensor([[0, 1], [1, 0]])  # node 2 isolated
    mask = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool)
    out = MUSE([4, 4], 3).eval()(xs, ei, mask)
    assert (out["alphas"][0, 2] == 1).all()  # modality 0 unobserved in node 2's receptive field


def test_propagation_preserves_properties():
    from muse.model import propagate_evidence
    torch.manual_seed(0)
    N, C, K = 40, 5, 4
    ei = torch.randint(0, N, (2, 150))
    E = torch.rand(K, N, C, dtype=torch.float64) * 5
    # commutes with the N-modal sum -> associativity / per-modality attribution
    assert torch.allclose(propagate_evidence(E.sum(0), ei), propagate_evidence(E, ei).sum(0), atol=1e-10)
    # vacuous identity: zero evidence stays zero
    assert propagate_evidence(torch.zeros(N, C, dtype=torch.float64), ei).abs().max() == 0
    # monotonicity: extra non-negative evidence never lowers total strength anywhere
    P1 = propagate_evidence(E[:2].sum(0), ei).sum(-1)
    P2 = propagate_evidence(E[:3].sum(0), ei).sum(-1)
    assert (P2 >= P1 - 1e-12).all()

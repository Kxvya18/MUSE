import torch

from muse3d.losses import loss_for
from muse3d.models import MODELS, per_sequence_features


def _batch():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 32, 32, 32)
    y = torch.randint(0, 4, (2, 32, 32, 32))
    mask = torch.tensor([[1, 0, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    return x, y, mask


def test_shapes_and_losses_are_finite():
    x, y, mask = _batch()
    for name, make in MODELS.items():
        m = make()
        out = m(x, mask)
        key = "alpha" if "alpha" in out else ("prob" if "prob" in out else "logits")
        assert out[key].shape == (2, 4, 32, 32, 32), name
        loss = loss_for(name)(out, y, anneal=0.5)
        loss.backward()
        assert torch.isfinite(loss), name


def test_absent_sequences_are_vacuous_and_ignored():
    x, _, mask = _batch()
    m = MODELS["MUSE"]().eval()
    out = m(x, mask)
    x2 = x.clone()
    x2[:, 1] = 1e4  # garbage in a sequence absent for every sample
    x2[1, 2:] = -1e4  # and in sequences absent for sample 1
    out2 = m(x2, mask)
    assert torch.allclose(out["alpha"], out2["alpha"], atol=1e-5)
    assert (out["alphas"][1] == 1).all() and (out["gammas"][1] == 0).all()


def test_fuse_matches_forward():
    x, _, mask = _batch()
    for name in ("MUSE", "HeMIS", "LateMean", "TMC", "RFNet", "mmFormer"):
        m = MODELS[name]().eval()
        with torch.no_grad():
            ref = m(x, mask)
            H = per_sequence_features(m.encoders, x, torch.ones_like(mask), m.F)
            fused = m.fuse(H, mask)
        key = next(k for k in ("alpha", "prob", "logits") if k in ref)
        assert torch.allclose(ref[key], fused[key], atol=1e-5), name


def test_new_baselines_ignore_absent_sequences():
    x, _, mask = _batch()
    for name in ("RFNet", "mmFormer"):
        m = MODELS[name]().eval()
        x2 = x.clone()
        x2[:, 1] = 1e4  # sequence 1 is absent for every sample
        with torch.no_grad():
            assert torch.allclose(m(x, mask)["logits"], m(x2, mask)["logits"], atol=1e-4), name


def test_muse2_vacuous_and_pattern_reuse():
    x, y, mask = _batch()
    for name in ("MUSE2-ctx", "MUSE2-ctx-ds"):
        m = MODELS[name]().eval()
        x2 = x.clone()
        x2[:, 1] = 1e4  # absent for every sample
        x2[1, 2:] = -1e4  # absent for sample 1
        with torch.no_grad():
            ref = m(x, mask)
            assert torch.allclose(ref["alpha"], m(x2, mask)["alpha"], atol=1e-4), name
            assert (ref["alphas"][1] == 1).all() and (ref["gammas"][1] == 0).all()
            enc = m.encode_all(x, torch.ones_like(mask))  # encode once, decode per pattern
            assert torch.allclose(m.decode_fuse(enc, mask)["alpha"], ref["alpha"], atol=1e-4), name
        m.train()
        out = m(x, mask)
        loss = loss_for(name)(out, y, anneal=0.5)
        loss.backward()
        assert torch.isfinite(loss)
        assert ("alpha_ds" in out) == name.endswith("-ds")

"""Voxel-level MUSE and matched missing-sequence baselines on one 3D U-Net backbone.

Every model takes x [B, 4, D, H, W] and mask [B, 4] (True = sequence acquired) and returns
either {"logits"} (softmax models) or {"alpha"} (evidential models), shape [B, C, D, H, W].

  MUSE      one U-Net per sequence -> evidential head -> vacuous mass for absent sequences
            -> per-voxel, per-class reliability gate -> sum fusion of discounted evidence
  UNet-MD   one U-Net on all 4 channels, absent ones zero-filled, trained with random
            sequence dropout (the standard "modality dropout" baseline); wider (base 32)
            so its size matches the 4-encoder models
  HeMIS     per-sequence U-Nets, features fused by mean/variance over present sequences,
            then a small head (Havaei et al., MICCAI 2016)
  LateMean  per-sequence U-Nets with softmax heads, probabilities averaged over present ones
  TMC       per-sequence evidential heads fused with Dempster's rule (Han et al., TPAMI 2022)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from muse.dempster import chain, to_dirichlet


def block(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01, True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01, True))


class UNet3D(nn.Module):
    """Plain 4-level 3D U-Net returning full-resolution features (base channels)."""

    def __init__(self, cin, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([block(cin, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])])
        self.down = nn.ModuleList([nn.Conv3d(c[i], c[i], 2, stride=2) for i in range(3)])
        self.up = nn.ModuleList([nn.ConvTranspose3d(c[i + 1], c[i], 2, stride=2) for i in range(3)])
        self.dec = nn.ModuleList([block(2 * c[i], c[i]) for i in range(3)])

    def forward(self, x):
        skips = []
        for i in range(3):
            x = self.enc[i](x)
            skips.append(x)
            x = self.down[i](x)
        x = self.enc[3](x)
        for i in reversed(range(3)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return x


def per_sequence_features(encoders, x, mask, feat):
    """Run encoder k only on the samples that acquired sequence k; absent -> zeros (unused)."""
    B = x.size(0)
    H = x.new_zeros(len(encoders), B, feat, *x.shape[2:])
    for k, enc in enumerate(encoders):
        idx = mask[:, k].nonzero(as_tuple=True)[0]
        if idx.numel():
            H[k, idx] = enc(x[idx, k:k + 1]).to(H.dtype)
    return H


class MUSE3D(nn.Module):
    def __init__(self, C=4, base=16, use_gate=True, combiner="sum"):
        super().__init__()
        self.C, self.F, self.use_gate, self.combiner = C, base, use_gate, combiner
        self.encoders = nn.ModuleList([UNet3D(1, base) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Conv3d(base, C, 1) for _ in range(4)])
        self.gates = nn.ModuleList([nn.Conv3d(2 * base, C, 1) for _ in range(4)])

    def forward(self, x, mask):
        return self.fuse(per_sequence_features(self.encoders, x, mask, self.F), mask)

    def fuse(self, H, mask):
        """H: per-sequence features [K, B, F, ...]; entries of absent sequences are ignored."""
        m = mask.t().float()[:, :, None, None, None, None]  # K, B, 1, 1, 1, 1
        alphas = torch.stack([F.softplus(self.heads[k](H[k]).float()) + 1 for k in range(4)])
        alphas = torch.where(m.bool(), alphas, torch.ones_like(alphas))  # vacuous mass
        ctx = (H * m).sum(0) / m.sum(0).clamp_min(1)  # mean feature over present sequences
        if self.use_gate:
            gammas = torch.stack([torch.sigmoid(self.gates[k](torch.cat([H[k], ctx], 1)).float()) for k in range(4)])
        else:
            gammas = torch.ones_like(alphas)
        gammas = gammas * m
        if self.combiner == "dempster":
            b, u = chain(((alphas - 1) * gammas + 1).movedim(2, -1), [0, 1, 2, 3], "dempster")
            alpha = to_dirichlet(b, u).movedim(-1, 1)
        else:
            alpha = ((alphas - 1) * gammas).sum(0) + 1
        return {"alpha": alpha, "alphas": alphas, "gammas": gammas, "mask": mask}


class SeqUNet(nn.Module):
    """3D U-Net split into encode / decode so a cross-sequence context can be injected at the
    bottleneck. decode() returns decoder features at 1/4, 1/2 and full resolution."""

    def __init__(self, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.c = c
        self.enc = nn.ModuleList([block(1, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])])
        self.down = nn.ModuleList([nn.Conv3d(c[i], c[i], 2, stride=2) for i in range(3)])
        self.mix = nn.Conv3d(2 * c[3], c[3], 1)  # own bottleneck + shared context -> bottleneck
        self.up = nn.ModuleList([nn.ConvTranspose3d(c[i + 1], c[i], 2, stride=2) for i in range(3)])
        self.dec = nn.ModuleList([block(2 * c[i], c[i]) for i in range(3)])

    def encode(self, x):
        skips = []
        for i in range(3):
            x = self.enc[i](x)
            skips.append(x)
            x = self.down[i](x)
        return skips, self.enc[3](x)

    def decode(self, skips, z, ctx):
        x = self.mix(torch.cat([z, ctx], 1))
        outs = []
        for i in reversed(range(3)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
            outs.append(x)
        return outs  # [1/4, 1/2, full]


class MUSE3Dv2(nn.Module):
    """MUSE with cross-sequence context: the bottlenecks of the *present* sequences are averaged
    into a shared context that every present sequence's decoder sees, so each source's evidence
    is informed by the others before evidential fusion. Absent sequences are still never
    encoded, are excluded from the context, and contribute vacuous mass (alpha = 1); the
    gate and the sum combiner are unchanged, so the three fusion properties still hold.
    With deep_sup=True the fused evidence is also produced at 1/2 and 1/4 resolution for
    deep supervision (training only; the output is the full-resolution evidence)."""

    def __init__(self, C=4, base=16, deep_sup=False):
        super().__init__()
        self.C, self.F, self.deep_sup = C, base, deep_sup
        self.nets = nn.ModuleList([SeqUNet(base) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Conv3d(base, C, 1) for _ in range(4)])
        self.gates = nn.ModuleList([nn.Conv3d(2 * base, C, 1) for _ in range(4)])
        ch = self.nets[0].c
        self.ds_heads = nn.ModuleList([nn.ModuleList([nn.Conv3d(ch[2], C, 1), nn.Conv3d(ch[1], C, 1)]) for _ in range(4)])

    def encode_all(self, x, mask):
        """Encoder outputs per sequence (only for samples that acquired it)."""
        enc = []
        for k, net in enumerate(self.nets):
            idx = mask[:, k].nonzero(as_tuple=True)[0]
            enc.append((idx, net.encode(x[idx, k:k + 1])) if idx.numel() else (idx, None))
        return enc

    def decode_fuse(self, enc, mask):
        B = mask.size(0)
        # shared context: mean bottleneck over the sequences present for each sample
        zs = [e[1][1] for e in enc if e[1] is not None]
        z0 = zs[0]
        ctx_sum = z0.new_zeros((B,) + z0.shape[1:])
        for k, (idx, e) in enumerate(enc):
            if e is not None:
                keep = mask[idx, k]
                ctx_sum[idx[keep]] += e[1][keep]
        ctx = ctx_sum / mask.sum(1).clamp_min(1).view(B, 1, 1, 1, 1).to(ctx_sum.dtype)
        feats = [[None] * 3 for _ in range(4)]
        shapes = None
        for k, (idx, e) in enumerate(enc):
            if e is None:
                continue
            keep = mask[idx, k]
            if not keep.any():
                continue
            rows = idx[keep]
            skips = [s[keep] for s in e[0]]
            outs = self.nets[k].decode(skips, e[1][keep], ctx[rows])
            feats[k] = (rows, outs)
            shapes = [o.shape for o in outs]
        m = mask.t().float()[:, :, None, None, None, None]
        def gather(level, width):  # stack per-sequence features at one decoder level -> K,B,width,...
            shape = shapes[level][2:]
            H = ctx.new_zeros((4, B, width) + tuple(shape))
            for k in range(4):
                if feats[k][0] is not None:
                    rows, outs = feats[k]
                    H[k, rows] = outs[level].to(H.dtype)
            return H
        H = gather(2, self.F)
        alphas = torch.stack([F.softplus(self.heads[k](H[k]).float()) + 1 for k in range(4)])
        alphas = torch.where(m.bool(), alphas, torch.ones_like(alphas))
        hctx = (H * m).sum(0) / m.sum(0).clamp_min(1)
        gammas = torch.stack([torch.sigmoid(self.gates[k](torch.cat([H[k], hctx], 1)).float()) for k in range(4)]) * m
        out = {"alpha": ((alphas - 1) * gammas).sum(0) + 1, "alphas": alphas, "gammas": gammas, "mask": mask}
        if self.deep_sup and self.training:
            ch = self.nets[0].c
            ds = []
            for j, (level, width) in enumerate([(0, ch[2]), (1, ch[1])]):
                Hs = gather(level, width)
                a = torch.stack([F.softplus(self.ds_heads[k][j](Hs[k]).float()) + 1 for k in range(4)])
                a = torch.where(m.bool(), a, torch.ones_like(a))
                ds.append(((a - 1).sum(0) + 1))
            out["alpha_ds"] = ds  # [1/4 resolution, 1/2 resolution]
        return out

    def forward(self, x, mask):
        return self.decode_fuse(self.encode_all(x, mask), mask)


class UNetMD(nn.Module):
    def __init__(self, C=4, base=32):
        super().__init__()
        self.net, self.out = UNet3D(4, base), nn.Conv3d(base, C, 1)

    def forward(self, x, mask):
        return {"logits": self.out(self.net(x * mask[:, :, None, None, None].to(x.dtype)))}


class HeMIS3D(nn.Module):
    def __init__(self, C=4, base=16):
        super().__init__()
        self.F = base
        self.encoders = nn.ModuleList([UNet3D(1, base) for _ in range(4)])
        self.head = nn.Sequential(block(2 * base, base), nn.Conv3d(base, C, 1))

    def forward(self, x, mask):
        return self.fuse(per_sequence_features(self.encoders, x, mask, self.F), mask)

    def fuse(self, H, mask):
        m = mask.t().float()[:, :, None, None, None, None]
        n = m.sum(0).clamp_min(1)
        mu = (H * m).sum(0) / n
        var = (((H - mu) ** 2) * m).sum(0) / n
        return {"logits": self.head(torch.cat([mu, var], 1))}


class LateMean3D(nn.Module):
    def __init__(self, C=4, base=16):
        super().__init__()
        self.F = base
        self.encoders = nn.ModuleList([UNet3D(1, base) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Conv3d(base, C, 1) for _ in range(4)])

    def forward(self, x, mask):
        return self.fuse(per_sequence_features(self.encoders, x, mask, self.F), mask)

    def fuse(self, H, mask):
        m = mask.t().float()[:, :, None, None, None, None]
        logits = torch.stack([self.heads[k](H[k]).float() for k in range(4)])
        prob = (torch.softmax(logits, 2) * m).sum(0) / m.sum(0)
        return {"prob": prob, "logits_k": logits, "mask": mask}


class TMC3D(nn.Module):
    def __init__(self, C=4, base=16):
        super().__init__()
        self.F = base
        self.encoders = nn.ModuleList([UNet3D(1, base) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Conv3d(base, C, 1) for _ in range(4)])

    def forward(self, x, mask):
        return self.fuse(per_sequence_features(self.encoders, x, mask, self.F), mask)

    def fuse(self, H, mask):
        m = mask.t().float()[:, :, None, None, None, None]
        alphas = torch.stack([F.softplus(self.heads[k](H[k]).float()) + 1 for k in range(4)])
        alphas = torch.where(m.bool(), alphas, torch.ones_like(alphas))  # absent = vacuous
        b, u = chain(alphas.movedim(2, -1), [0, 1, 2, 3], "dempster")
        return {"alpha": to_dirichlet(b, u).movedim(-1, 1), "alphas": alphas, "mask": mask}


class RFNet3D(nn.Module):
    """RFNet-style region-aware fusion (Ding et al., ICCV 2021), basic version on our backbone:
    for each tumour region r, voxel-wise softmax weights over the *present* sequences
    combine their features; the region features are concatenated for the output head.
    Per-sequence auxiliary heads are trained as in RFNet."""

    def __init__(self, C=4, base=16, regions=3):
        super().__init__()
        self.F, self.R = base, regions
        self.encoders = nn.ModuleList([UNet3D(1, base) for _ in range(4)])
        self.score = nn.ModuleList([nn.Conv3d(base, regions, 1) for _ in range(4)])
        self.head = nn.Sequential(block(regions * base, base), nn.Conv3d(base, C, 1))
        self.aux = nn.ModuleList([nn.Conv3d(base, C, 1) for _ in range(4)])

    def forward(self, x, mask):
        return self.fuse(per_sequence_features(self.encoders, x, mask, self.F), mask)

    def fuse(self, H, mask):
        m = mask.t()[:, :, None, None, None, None]  # K, B, 1, 1, 1, 1
        s = torch.stack([self.score[k](H[k]).float() for k in range(4)])  # K, B, R, ...
        w = torch.softmax(s.masked_fill(~m, -1e4), 0)  # weights over present sequences only
        fused = torch.cat([(w[:, :, r:r + 1] * H.float()).sum(0) for r in range(self.R)], 1)
        aux = torch.stack([self.aux[k](H[k]).float() for k in range(4)])
        return {"logits": self.head(fused.to(H.dtype)), "aux_logits": aux, "mask": mask}


class MMFormer3D(nn.Module):
    """mmFormer-style fusion (Zhang et al., MICCAI 2022), basic version on our backbone:
    per-sequence features are pooled to a coarse grid, a small Transformer attends across
    sequences and space with absent sequences masked out, and the result (averaged over
    present sequences, upsampled) is added to the mean full-resolution feature."""

    def __init__(self, C=4, base=16, d=64, pool=16, layers=2, heads=4):
        super().__init__()
        self.F, self.pool = base, pool
        self.encoders = nn.ModuleList([UNet3D(1, base) for _ in range(4)])
        self.inp, self.out = nn.Linear(base, d), nn.Linear(d, base)
        self.seq_embed = nn.Parameter(torch.zeros(4, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(block(base, base), nn.Conv3d(base, C, 1))

    def forward(self, x, mask):
        return self.fuse(per_sequence_features(self.encoders, x, mask, self.F), mask)

    def fuse(self, H, mask):
        K, B, F_ = H.shape[:3]
        size = H.shape[3:]
        m = mask.t().float()[:, :, None, None, None, None]
        coarse = torch.stack([F.avg_pool3d(H[k].float(), self.pool) for k in range(K)])  # K,B,F,d,h,w
        g = coarse.shape[3:]
        n = g[0] * g[1] * g[2]
        tok = coarse.flatten(3).permute(1, 0, 3, 2)  # B, K, n, F
        tok = self.inp(tok) + self.seq_embed[None, :, None, :]
        tok = tok.reshape(B, K * n, -1)
        pad = (~mask).repeat_interleave(n, dim=1)  # True = ignore (absent sequence)
        tok = self.tf(tok, src_key_padding_mask=pad).reshape(B, K, n, -1)
        keep = mask.float()[:, :, None, None]
        tok = (tok * keep).sum(1) / keep.sum(1).clamp_min(1)  # mean over present sequences
        ctx = self.out(tok).permute(0, 2, 1).reshape(B, F_, *g)
        ctx = F.interpolate(ctx, size=size, mode="trilinear", align_corners=False)
        local = ((H.float() * m).sum(0) / m.sum(0).clamp_min(1))
        return {"logits": self.head((local + ctx).to(H.dtype)), "mask": mask}


MODELS = {
    "MUSE": lambda: MUSE3D(),
    "UNet-MD": lambda: UNetMD(),
    "HeMIS": lambda: HeMIS3D(),
    "LateMean": lambda: LateMean3D(),
    "TMC": lambda: TMC3D(),
    "MUSE2-ctx": lambda: MUSE3Dv2(),
    "MUSE2-ctx-ds": lambda: MUSE3Dv2(deep_sup=True),
    "RFNet": lambda: RFNet3D(),
    "mmFormer": lambda: MMFormer3D(),
    "MUSE-noGate": lambda: MUSE3D(use_gate=False),
    "MUSE-Dempster": lambda: MUSE3D(combiner="dempster"),
}


def probs_and_uncertainty(out):
    """Expected class probabilities [B,C,...] and a per-voxel uncertainty [B,...]."""
    if "alpha" in out:
        a = out["alpha"]
        S = a.sum(1, keepdim=True)
        return a / S, (a.size(1) / S).squeeze(1)
    p = out["prob"] if "prob" in out else torch.softmax(out["logits"].float(), 1)
    ent = -(p * p.clamp_min(1e-8).log()).sum(1) / torch.log(torch.tensor(float(p.size(1))))
    return p, ent

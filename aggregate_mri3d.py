"""Tables for the voxel-level MRI experiment (runs_final/<model>_s0/test.json).

Only patients present for every model are used, so all methods are compared on the same
cases. Per-patient scores are averaged over the 15 sequence patterns; significance is a
paired Wilcoxon signed-rank test of MUSE against each other model over patients.

usage: python aggregate_mri3d.py [--runs runs_final] [--out results/mri3d_tables]
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ORDER = ["MUSE", "UNet-MD", "HeMIS", "LateMean", "TMC", "RFNet", "mmFormer", "MUSE-noGate", "MUSE-Dempster", "MUSE2-ctx-ds"]
VARIANT = lambda m: m.startswith("MUSE-") or m.startswith("MUSE2")  # ablations / variants: not ranked as baselines
REG = ["WT", "TC", "ET"]


def load(runs):
    R = {}
    for m in ORDER:
        f = os.path.join(runs, f"{m}_s0", "test.json")
        if os.path.exists(f):
            R[m] = json.load(open(f))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_final")
    ap.add_argument("--out", default="results/mri3d_tables")
    ap.add_argument("--exclude", default="", help="comma-separated models to leave out (e.g. still running)")
    a = ap.parse_args()
    R = {m: v for m, v in load(a.runs).items() if m not in a.exclude.split(",")}
    common = sorted(set.intersection(*[set(v) for v in R.values()]))
    pats = list(next(iter(R.values()))[common[0]].keys())
    pats.sort(key=lambda p: (-p.count("+"), p))
    models = [m for m in ORDER if m in R]

    def per_patient(m, key):  # [patients, patterns]
        return np.array([[R[m][c][p][key] for p in pats] for c in common], dtype=float)

    md = [f"# MRI results (MSD Task01, {len(common)} held-out patients common to all models, 15 sequence patterns)\n"]
    md.append("## Overall (mean over patterns; ± = std over patients)\n")
    md.append("| Method | Dice WT | Dice TC | Dice ET | Dice mean | HD95 WT | HD95 TC | HD95 ET | ECE ↓ | AUROC err | u bnd/healthy | p vs MUSE |")
    md.append("|" + "---|" * 12)
    summary = {}
    musemean = per_patient("MUSE", "dice_WT").mean(1) * 0
    if "MUSE" in R:
        musemean = np.mean([per_patient("MUSE", f"dice_{r}").mean(1) for r in REG], 0)
    best = {}
    for m in models:
        d = {r: per_patient(m, f"dice_{r}") for r in REG}
        h = {r: per_patient(m, f"hd95_{r}") for r in REG}
        mean_pp = np.mean([d[r].mean(1) for r in REG], 0)
        row = {f"dice_{r}": d[r].mean() for r in REG}
        row.update({f"hd95_{r}": h[r].mean() for r in REG})
        row["dice_mean"] = mean_pp.mean()
        row["ece"] = np.nanmean(per_patient(m, "ece"))
        row["auroc"] = np.nanmean(per_patient(m, "auroc_err"))
        row["u_ratio"] = np.nanmean(per_patient(m, "u_boundary")) / np.nanmean(per_patient(m, "u_healthy"))
        row["std"] = {f"dice_{r}": d[r].mean(1).std() for r in REG} | {"dice_mean": mean_pp.std()}
        row["p"] = (wilcoxon(musemean, mean_pp).pvalue if m != "MUSE" and np.any(musemean != mean_pp) else None)
        summary[m] = row
    for k, better in [("dice_WT", max), ("dice_TC", max), ("dice_ET", max), ("dice_mean", max), ("hd95_WT", min),
                      ("hd95_TC", min), ("hd95_ET", min), ("ece", min), ("auroc", max)]:
        best[k] = better([m for m in models if not VARIANT(m)], key=lambda m: summary[m][k])
    fmt = lambda m, k, v: f"**{v}**" if best.get(k) == m else v
    for m in models:
        s = summary[m]
        cells = [fmt(m, f"dice_{r}", f"{s[f'dice_{r}']:.3f} ± {s['std'][f'dice_{r}']:.3f}") for r in REG]
        cells.append(fmt(m, "dice_mean", f"{s['dice_mean']:.3f} ± {s['std']['dice_mean']:.3f}"))
        cells += [fmt(m, f"hd95_{r}", f"{s[f'hd95_{r}']:.1f}") for r in REG]
        cells += [fmt(m, "ece", f"{s['ece']:.4f}"), fmt(m, "auroc", f"{s['auroc']:.3f}"), f"{s['u_ratio']:.2f}"]
        cells.append("—" if s["p"] is None else f"{s['p']:.1e}")
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    md.append("\nBold = best among MUSE and the baselines (ablations excluded). p: paired Wilcoxon over patients "
              "(per-patient Dice averaged over WT/TC/ET and the 15 patterns).\n")
    md.append("## Mean Dice (WT/TC/ET averaged) per available-sequence pattern\n")
    md.append("| Sequences | " + " | ".join(models) + " |")
    md.append("|" + "---|" * (len(models) + 1))
    per_pat = {}
    for j, p in enumerate(pats):
        vals = [np.mean([per_patient(m, f"dice_{r}")[:, j].mean() for r in REG]) for m in models]
        per_pat[p] = dict(zip(models, vals))
        main_idx = [i for i, m in enumerate(models) if not VARIANT(m)]
        b = max(main_idx, key=lambda i: vals[i])
        md.append(f"| {p} | " + " | ".join((f"**{v:.3f}**" if i == b else f"{v:.3f}") for i, v in enumerate(vals)) + " |")
    wins = sum(max([m for m in models if not VARIANT(m)], key=lambda m: per_pat[p][m]) == "MUSE" for p in pats)
    md.append(f"\nMUSE is best in {wins} of {len(pats)} sequence patterns.\n")
    if "MUSE" in R and any("contrib" in R["MUSE"][common[0]][p] for p in pats):
        md.append("## MUSE per-sequence contribution on tumour voxels (all four sequences available)\n")
        full = "FLAIR+T1+T1gd+T2"
        c = np.array([R["MUSE"][cc][full]["contrib"] for cc in common])
        md.append("| FLAIR | T1 | T1gd | T2 |\n|---|---|---|---|\n| " + " | ".join(f"{v:.3f}" for v in c.mean(0)) + " |")
    nonfinite = {m: int(sum(R[m][c][p].get("nonfinite_voxels", 0) for c in common for p in pats)) for m in models}
    md.append(f"\nNon-finite voxels encountered (neutralised): {nonfinite}\n")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "mri3d_results.md"), "w").write("\n".join(md))
    json.dump({"patients": len(common), "summary": {m: {k: v for k, v in s.items()} for m, s in summary.items()},
               "per_pattern": per_pat}, open(os.path.join(a.out, "mri3d_summary.json"), "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()

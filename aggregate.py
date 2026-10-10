"""Aggregate final results into journal tables (LaTeX + Markdown).

Graph benchmarks (results/final): per dataset x model, mean +- std over seeds of
accuracy / macro-F1 / ECE / NLL / error-detection AUROC at each missing rate, plus
averages over rates and over the 15 fixed patterns. Paired tests vs MUSE (same seeds
share data splits and missing masks): Wilcoxon signed-rank over seeds x rates, and a
paired t-test over seeds of the rate-averaged metric. Best per column in bold.

MRI (results/brats): Dice WT/TC/ET per missing-sequence pattern and averaged, node
ECE / AUROC, boundary-uncertainty statistics.
"""
from __future__ import annotations

import collections
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon, ttest_rel

OUT = "results/tables"
RATES = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]
ORDER = ["MUSE", "GCN-zero", "GAT-zero", "APPNP-zero", "MMGL-style", "TMC", "ECD", "EGNN", "HeMIS"]
LOWER_BETTER = {"ece", "nll"}


def fmt(m, s, pct=True, bold=False):
    t = f"{100*m:.1f}$\\pm${100*s:.1f}" if pct else f"{m:.3f}$\\pm${s:.3f}"
    return f"\\textbf{{{t}}}" if bold else t


def load_graph():
    R = collections.defaultdict(list)
    for f in glob.glob("results/final/*.json"):
        r = json.load(open(f))
        R[(r["dataset"], r["model"])].append(r)
    for k in R:
        R[k].sort(key=lambda r: r["seed"])
    return R


def per_seed(runs, metric, rates=RATES):
    """[n_seeds, n_rates]"""
    return np.array([[r["rates"][q][metric] for q in rates] for r in runs])


def graph_tables(R):
    os.makedirs(OUT, exist_ok=True)
    datasets = sorted({d for d, _ in R}, key=lambda d: ["Cora", "CiteSeer", "Photo"].index(d))
    md, tex = [], []
    summary = {}
    for ds in datasets:
        models = [m for m in ORDER if (ds, m) in R]
        ablations = sorted(m for (d, m) in R if d == ds and m not in ORDER)
        # main table: accuracy at selected rates + mean, ECE mean, AUROC mean
        cols = ["0.0", "0.3", "0.5", "0.7"]
        stats = {}
        for m in models + ablations:
            acc, ece, au, f1 = (per_seed(R[(ds, m)], k) for k in ("acc", "ece", "auroc_err", "f1"))
            pat = np.array([[r["patterns"][p]["acc"] for p in r["patterns"]] for r in R[(ds, m)]])
            stats[m] = {"acc": acc, "ece": ece, "auroc": au, "f1": f1, "pat": pat, "n": len(R[(ds, m)])}
        best = {}
        for c in cols:
            best[c] = max(models, key=lambda m: stats[m]["acc"][:, RATES.index(c)].mean())
        for key, fn in (("acc_mean", lambda m: stats[m]["acc"].mean()), ("f1_mean", lambda m: stats[m]["f1"].mean()),
                        ("pat_mean", lambda m: stats[m]["pat"].mean()), ("auroc", lambda m: np.nanmean(stats[m]["auroc"]))):
            best[key] = max(models, key=fn)
        best["ece"] = min(models, key=lambda m: stats[m]["ece"].mean())
        tex.append(f"% ---- {ds}\n\\begin{{tabular}}{{l" + "c" * 9 + "}\n\\toprule")
        tex.append("Method & " + " & ".join(f"Acc@{int(float(c)*100)}\\%" for c in cols) +
                   " & Acc (avg) & F1 (avg) & Acc (15 pat.) & ECE (avg)$\\downarrow$ & AUROC$_{err}$ \\\\\n\\midrule")
        md.append(f"\n### {ds}  (mean ± std over {stats[models[0]]['n']} seeds; test set)\n")
        md.append("| Method | " + " | ".join(f"Acc@{int(float(c)*100)}%" for c in cols) +
                  " | Acc avg | F1 avg | Acc 15 patterns | ECE avg ↓ | AUROC err | p (acc vs MUSE) |")
        md.append("|" + "---|" * 11)
        summary[ds] = {}
        for m in models + ["---"] + ablations:
            if m == "---":
                tex.append("\\midrule")
                continue
            S = stats[m]
            seedmean = S["acc"].mean(1)
            if m != "MUSE" and "MUSE" in stats and stats["MUSE"]["acc"].shape == S["acc"].shape:
                d = (stats["MUSE"]["acc"] - S["acc"]).ravel()
                p_w = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                p_t = ttest_rel(stats["MUSE"]["acc"].mean(1), seedmean).pvalue
                ptxt = f"{p_w:.1e} / {p_t:.1e}"
            else:
                ptxt = "—"
            cells = [fmt(S["acc"][:, RATES.index(c)].mean(), S["acc"][:, RATES.index(c)].std(), bold=best[c] == m) for c in cols]
            cells += [fmt(S["acc"].mean(), seedmean.std(), bold=best["acc_mean"] == m),
                      fmt(S["f1"].mean(), S["f1"].mean(1).std(), bold=best["f1_mean"] == m),
                      fmt(S["pat"].mean(), S["pat"].mean(1).std(), bold=best["pat_mean"] == m),
                      fmt(S["ece"].mean(), S["ece"].mean(1).std(), bold=best["ece"] == m),
                      fmt(np.nanmean(S["auroc"]), np.nanstd(np.nanmean(S["auroc"], 1)), bold=best["auroc"] == m)]
            tex.append(f"{m} & " + " & ".join(cells) + " \\\\")
            md.append(f"| {m} | " + " | ".join(c.replace("$\\pm$", " ± ").replace("\\textbf{", "**").replace("}", "**") if "textbf" in c
                                               else c.replace("$\\pm$", " ± ") for c in cells) + f" | {ptxt} |")
            summary[ds][m] = {"acc_avg": float(S["acc"].mean()), "ece_avg": float(S["ece"].mean()),
                              "acc_by_rate": S["acc"].mean(0).tolist(), "ece_by_rate": S["ece"].mean(0).tolist(),
                              "auroc_by_rate": np.nanmean(S["auroc"], 0).tolist(), "acc_patterns": float(S["pat"].mean())}
        tex.append("\\bottomrule\n\\end{tabular}")
    open(f"{OUT}/graph_tables.tex", "w").write("\n".join(tex))
    open(f"{OUT}/graph_tables.md", "w").write("\n".join(md))
    json.dump(summary, open(f"{OUT}/graph_summary.json", "w"), indent=1)


def mri_tables():
    R = collections.defaultdict(list)
    for f in glob.glob("results/brats/final_*.json"):
        r = json.load(open(f))
        R[r["model"]].append(r)
    if not R:
        return
    pats = list(next(iter(R.values()))[0]["test"].keys())
    models = [m for m in ORDER if m in R] + sorted(m for m in R if m not in ORDER)
    md = ["\n### MRI (MSD Task01, 98 held-out patients; mean over seeds)\n",
          "| Method | Dice WT (avg 15) | Dice TC (avg 15) | Dice ET (avg 15) | Dice mean (all 4 seq.) | node ECE ↓ | AUROC err | u boundary / interior |",
          "|" + "---|" * 8]
    tex = ["\\begin{tabular}{lccccccc}\n\\toprule",
           "Method & WT & TC & ET & Mean (full) & ECE$\\downarrow$ & AUROC$_{err}$ & $u_{bnd}/u_{int}$ \\\\\n\\midrule"]
    per_pattern = {}
    rows = {}
    for m in models:
        runs = R[m]
        d = {k: np.mean([[r["test"][p]["dice"][k] for p in pats] for r in runs], 0) for k in ("WT", "TC", "ET")}
        full = np.mean([np.mean(list(r["test"]["FLAIR+T1+T1gd+T2"]["dice"].values())) for r in runs])
        ece = np.mean([[r["test"][p]["node"]["ece"] for p in pats] for r in runs])
        au = np.nanmean([[r["test"][p]["node"]["auroc_err"] for p in pats] for r in runs])
        ub = np.nanmean([[r["test"][p]["boundary"]["u_boundary"] for p in pats] for r in runs])
        ui = np.nanmean([[r["test"][p]["boundary"]["u_interior"] for p in pats] for r in runs])
        rows[m] = (d["WT"].mean(), d["TC"].mean(), d["ET"].mean(), full, ece, au, ub / max(ui, 1e-9))
        per_pattern[m] = {p: {k: float(d[k][i]) for k in d} for i, p in enumerate(pats)}
    best = [max(models, key=lambda m: rows[m][i]) if i not in (4,) else min(models, key=lambda m: rows[m][i]) for i in range(6)]
    for m in models:
        v = rows[m]
        cells = [(f"\\textbf{{{v[i]:.3f}}}" if best[i] == m else f"{v[i]:.3f}") for i in range(6)] + [f"{v[6]:.2f}"]
        tex.append(f"{m} & " + " & ".join(cells) + " \\\\")
        md.append(f"| {m} | " + " | ".join(c.replace("\\textbf{", "**").replace("}", "**") for c in cells) + " |")
    tex.append("\\bottomrule\n\\end{tabular}")
    # per-pattern Dice table (mean of WT/TC/ET)
    md.append("\n#### Mean Dice per available-sequence pattern\n")
    md.append("| Pattern | " + " | ".join(models) + " |")
    md.append("|" + "---|" * (len(models) + 1))
    for p in pats:
        vals = [np.mean(list(per_pattern[m][p].values())) for m in models]
        b = int(np.argmax(vals))
        md.append(f"| {p} | " + " | ".join((f"**{v:.3f}**" if i == b else f"{v:.3f}") for i, v in enumerate(vals)) + " |")
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/mri_tables.tex", "w").write("\n".join(tex))
    open(f"{OUT}/mri_tables.md", "w").write("\n".join(md))
    json.dump(per_pattern, open(f"{OUT}/mri_per_pattern.json", "w"), indent=1)


if __name__ == "__main__":
    R = load_graph()
    if R:
        graph_tables(R)
    mri_tables()
    print("tables written to", OUT)

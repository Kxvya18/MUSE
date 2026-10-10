"""Validation-only design study #3 (test set never evaluated)."""
import sys, time
from run_graph import run
ds = sys.argv[1]
base = dict(lr=1e-2, wd=5e-4, dropout=0.6, hidden=64)
variants = {"A gat+prop aux-raw": ("MUSE", {"aux_src": "raw"}),
            "B gat+prop aux-prop": ("MUSE", {}),
            "C mlp+prop aux-prop": ("MUSE-MLP", {})}
for name, (mn, extra) in variants.items():
    vs = []
    for s in (0, 1):
        t = time.time(); vs.append(run(ds, mn, s, {**base, **extra})["val"])
    print(f"[design3] {ds} {name:22s} val={sum(vs)/2:.4f} {[round(v,4) for v in vs]} ({time.time()-t:.0f}s/run)", flush=True)

"""Validation-only design study (test set never evaluated)."""
import json, sys, time
from run_graph import run
ds = sys.argv[1]
base = dict(lr=1e-2, wd=5e-4, dropout=0.3, hidden=64)
variants = {"MUSE(neigh)": ("MUSE", {}), "MUSE(neigh)+aux1": ("MUSE", {"aux": 1.0}),
            "MUSE(neigh)+aux0.5": ("MUSE", {"aux": 0.5}), "MUSE(self)+aux1": ("MUSE-selfEvidence", {"aux": 1.0})}
for name, (mn, extra) in variants.items():
    vs = []
    for s in (0, 1):
        t = time.time(); r = run(ds, mn, s, {**base, **extra}); vs.append(r["val"])
    print(f"[design] {ds} {name:22s} val={sum(vs)/2:.4f} {vs} ({time.time()-t:.0f}s/run)", flush=True)

"""Validation-only design study #2 (test set never evaluated)."""
import sys, time
from run_graph import run
ds = sys.argv[1]
base = dict(lr=1e-2, wd=5e-4, dropout=0.3, hidden=128)
variants = {"ref": {}, "exp-evidence": {"act": "exp"}, "digamma-CE": {"ce": "digamma"},
            "no-KL": {"lambda1": 0.0}, "aux2": {"aux": 2.0}}
for name, extra in variants.items():
    vs = []
    for s in (0, 1):
        t = time.time(); vs.append(run(ds, "MUSE", s, {**base, **extra})["val"])
    print(f"[design2] {ds} {name:14s} val={sum(vs)/2:.4f} {[round(v,4) for v in vs]} ({time.time()-t:.0f}s/run)", flush=True)

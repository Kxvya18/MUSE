"""Preview MRI run (one config, one seed) so qualitative panels exist early.
Not used for any reported number; final results come from run_brats.py tune/final."""
import json, os, time, torch
from run_brats import train, test
cfg = dict(lr=1e-3, wd=5e-4, dropout=0.2, hidden=64)
os.makedirs("results/brats_preview", exist_ok=True); os.makedirs("checkpoints", exist_ok=True)
log = open("results/brats_preview/log.txt", "a")
for mn in ["MUSE", "GAT-zero"]:
    t = time.time()
    model, v = train(mn, cfg, 100, "data/brats_graphs_c4", epochs=12, patience=4, log=log)
    torch.save(model.state_dict(), f"checkpoints/brats_{mn}_100.pt")
    r = {"model": mn, "seed": 100, "cfg": cfg, "val": v, "test": test(model, "data/brats_graphs_c4"), "time": time.time() - t}
    json.dump(r, open(f"results/brats_preview/final_{mn}_100.json", "w"), indent=1)
    full = r["test"]["FLAIR+T1+T1gd+T2"]["dice"]
    print(f"[preview] {mn} val_f1={v:.3f} full Dice WT/TC/ET={full['WT']:.3f}/{full['TC']:.3f}/{full['ET']:.3f} ({r['time']/60:.0f} min)", flush=True)

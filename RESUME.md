# MUSE — where we paused (2026-10-07, ~22:20)

## State
- Code: `muse/` (model, baselines, dempster, data, metrics), `run_graph.py` (tune/select/final),
  `run_brats.py`, `preprocess_brats.py`, `analysis_order.py`, `tests/` (9 passing).
- Cora validation grid (16 configs x 2 seeds), finished/partial in `results/tune/`:
  GAT-zero 0.7933 (done), GCN-zero 0.7880 (done), APPNP-zero ~0.805 (partial), TMC 0.785 (partial),
  HeMIS 0.777 (partial), MMGL-style/ECD/EGNN (partial or not started).
- MUSE (validation, Cora), history of design changes (all decided on validation only):
  spec version 0.696 -> neighbourhood evidence 0.718 -> + per-modality aux loss 0.770
  -> best grid config (no propagation) 0.7913 (`results/tune_noprop/`)
  -> + graph evidence propagation ~0.79 at dropout 0.3 (one config).
  MUSE-MLP encoder: 0.744 (rejected).
- Was running: design study 3 (`design_study3.py Cora`) comparing per-modality loss on raw vs
  propagated evidence (GATv2) and MLP encoder; no result yet.
- MRI: MSD Task01 extracted at `data/Task01_BrainTumour` (484 cases). Cube size 4 chosen
  (oracle Dice WT/TC/ET 0.925/0.901/0.783). Preprocessing started: 10/484 graphs done in
  `data/brats_graphs_c4/` (resumable — skips finished cases).

## Resume
```bash
cd ~/Downloads/MUSE && source .venv/bin/activate
nohup python design_study3.py Cora > logs_design3_cora.txt 2>&1 &
nohup python run_graph.py tune --datasets Cora --models GAT-zero,GCN-zero,TMC,MMGL-style,ECD,EGNN > logs_tune_cora_base.txt 2>&1 &
nohup python run_graph.py tune --datasets Cora --models HeMIS > logs_tune_cora_hemis.txt 2>&1 &
nohup python run_graph.py tune --datasets Cora --models APPNP-zero > logs_tune_cora_appnp.txt 2>&1 &
nohup python preprocess_brats.py --cube 4 --workers 5 > logs_preprocess.txt 2>&1 &
```
Then: pick the MUSE variant from design study 3, run the MUSE grid, and repeat for
CiteSeer/Photo; then `select` and `final` (10 seeds), then MRI tune/final.
Bar to beat on Cora validation: APPNP-zero ~0.805.

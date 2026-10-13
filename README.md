# MUSE — Modality-Universal Selective Evidence Fusion

Uncertainty-aware multi-modal classification/segmentation under arbitrary missing
modalities, without synthesis or imputation. Each available modality yields Dirichlet
evidence; missing modalities contribute the vacuous mass (alpha = 1); a contextual gate
discounts evidence per modality and class; evidence is fused by cumulative (sum) fusion and,
on graphs, propagated with personalised PageRank.

## Layout
- `muse/model.py` — MUSE (encoders, evidential heads, reliability gate, N-modal combiner,
  evidence propagation, loss)
- `muse/baselines.py` — GCN/GAT/APPNP with zero-imputation, MMGL-style, TMC, ECD, EGNN, HeMIS
- `muse/dempster.py` — Dempster, Yager, PCR5, averaging rules (order-dependence study)
- `run_graph.py` — Cora/CiteSeer/Amazon-Photo: `tune` (validation only) -> `select` -> `final` (10 seeds)
- `muse3d/`, `run_mri3d.py` — voxel-level MUSE on brain-tumour MRI (per-sequence 3D U-Nets, evidential fusion) vs UNet-MD, HeMIS, LateMean, TMC
- `colab/MUSE_colab.ipynb` — run everything on a free Colab T4 (resumable via Google Drive; opens result PRs)
- `preprocess_brats.py`, `run_brats.py` — earlier supervoxel-graph MRI variant (superseded)
- `analysis_order.py` — empirical order-dependence of fusion rules
- `tests/` — formal properties (associativity, vacuous identity, monotone uncertainty, no leakage)

## Setup
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torch_geometric numpy scipy scikit-learn scikit-image nibabel matplotlib pandas tqdm pytest
python -m pytest -q tests
```
Data: Planetoid/Amazon download automatically; MSD Task01_BrainTumour goes in `data/`.

## Protocol
All design and hyperparameter decisions are made on validation data only; every model gets
the same 16-config grid; final numbers are 10 seeds disjoint from tuning seeds.

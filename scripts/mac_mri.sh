#!/bin/bash
# MRI experiment on the Mac (Apple GPU): 2 mm training volumes, full-resolution scoring.
# Same settings for every model. Progress: logs_mac_mri.txt ; per-model logs in runs_mac/.
cd "$(dirname "$0")/.."
source .venv/bin/activate
export GH_TOKEN=$(gh auth token)
L=logs_mac_mri.txt
echo "[$(date '+%m-%d %H:%M')] prep (2 mm)" >> $L
python run_mri3d.py prep --src data/Task01_BrainTumour --data data/mri3d_2mm --factor 2 --workers 4 >> $L 2>&1
echo "[$(date '+%m-%d %H:%M')] training queue" >> $L
python kaggle/run_queue.py --runs runs_mac --data data/mri3d_2mm \
  --extra "--iters 4000 --patch 64 --batch 2 --val_every 500 --val_cases 24 --test_patch 96 --test_stride 64" >> $L 2>&1
echo "[$(date '+%m-%d %H:%M')] MAC MRI DONE" >> $L

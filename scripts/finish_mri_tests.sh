#!/bin/bash
# preprocess the 8 patients fetched separately, then resume MUSE and TMC tests (all local)
cd "$(dirname "$0")/.."
source .venv/bin/activate
L=logs_finish_tests.txt
echo "[$(date '+%m-%d %H:%M')] preprocessing fetched patients" >> $L
python - >> $L 2>&1 <<'PY'
import glob, os
from muse3d.data import preprocess_case
for f in sorted(glob.glob("data/raw_missing/imagesTr/*.nii.gz")):
    print("preprocessed", preprocess_case((f, "data/mri3d_test1mm", 1)), flush=True)
PY
for m in MUSE TMC; do
  echo "[$(date '+%m-%d %H:%M')] testing $m" >> $L
  python run_mri3d.py test --model $m --seed 0 --data data/mri3d_test1mm --out runs_final >> $L 2>&1
done
echo "[$(date '+%m-%d %H:%M')] FINISH TESTS DONE" >> $L

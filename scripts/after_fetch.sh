#!/bin/bash
# wait for the archive fetch, preprocess the 8 patients it delivered, then finish the tests
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/fetch_from_msd_tar.py data/raw_missing BRATS_316 BRATS_439 BRATS_030 BRATS_426 BRATS_185 BRATS_290 BRATS_464 BRATS_096 >> logs_finish_tests.txt 2>&1
python - <<'PY' >> logs_finish_tests.txt 2>&1
import glob, os
from muse3d.data import preprocess_case
for f in sorted(glob.glob("data/raw_missing/imagesTr/*.nii.gz")):
    if os.path.exists(f.replace("imagesTr", "labelsTr")):
        print("preprocessed", preprocess_case((f, "data/mri3d_test1mm", 1)), flush=True)
    else:
        print("label missing for", f, flush=True)
PY
bash scripts/finish_tests.sh

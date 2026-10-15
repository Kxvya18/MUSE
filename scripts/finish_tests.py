"""Finish MRI tests on the Mac for models whose Colab/Kaggle test stopped early.

Preprocesses only the 98 held-out test patients at full 1 mm resolution (same split as
run_mri3d.py prep: sorted cases, permutation seed 0, 70/10/20), then resumes each test.
"""
import glob, json, os, shutil, subprocess, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from muse3d.data import preprocess_case

SRC = os.path.expanduser("~/Library/CloudStorage/OneDrive-ManipalAcademyofHigherEducation/MUSE_data/Task01_BrainTumour")
OUT = "data/mri3d_test1mm"
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC, "imagesTr", "BRATS_*.nii.gz")))
cases = sorted(os.path.basename(f).replace(".nii.gz", "") for f in files)
perm = np.random.default_rng(0).permutation(len(cases))
n_tr, n_va = int(0.7 * len(cases)), int(0.1 * len(cases))
split = {"train": [cases[i] for i in perm[:n_tr]], "val": [cases[i] for i in perm[n_tr:n_tr + n_va]],
         "test": [cases[i] for i in perm[n_tr + n_va:]]}
json.dump(split, open(os.path.join(OUT, "split.json"), "w"), indent=1)
LOCAL = os.path.join(tempfile.gettempdir(), "muse_raw")


def fetch(sub, case):
    """Copy one scan out of OneDrive (online-only files download on read; retry on timeouts)."""
    dst = os.path.join(LOCAL, sub, case + ".nii.gz")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(8):
        try:
            shutil.copyfile(os.path.join(SRC, sub, case + ".nii.gz"), dst)
            if os.path.getsize(dst) > 0:
                return dst
        except OSError:
            pass
        time.sleep(15)
    raise RuntimeError(f"could not fetch {sub}/{case} from OneDrive")


for i, c in enumerate(split["test"]):
    if not os.path.exists(os.path.join(OUT, c + ".npz")):
        img = fetch("imagesTr", c)
        fetch("labelsTr", c)
        preprocess_case((img, OUT, 1))
        shutil.rmtree(LOCAL, ignore_errors=True)
    if i % 10 == 0:
        print(f"preprocessed {i}/{len(split['test'])}", flush=True)
for model in sys.argv[1:]:
    print("testing", model, flush=True)
    subprocess.run([sys.executable, "run_mri3d.py", "test", "--model", model, "--seed", "0",
                    "--data", OUT, "--out", "runs_final"], check=False)
print("FINISH TESTS DONE", flush=True)

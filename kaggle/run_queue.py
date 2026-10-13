"""Train + test MRI models on every visible GPU in parallel (Kaggle gives 2 x T4).

Each finished model is pushed to GitHub straight away (results PR), so a session that
hits Kaggle's 12-hour limit loses at most the models still running; those resume from
their last checkpoint in the next session.

usage: python kaggle/run_queue.py --runs /kaggle/working/runs --data /tmp/mri3d
"""
import argparse
import os
import subprocess
import sys
import time

import torch

QUEUE = ["MUSE", "UNet-MD", "HeMIS", "TMC", "LateMean", "MUSE-noGate", "MUSE-Dempster"]


def is_done(runs, model, seed):
    return os.path.exists(os.path.join(runs, f"{model}_s{seed}", "test_done"))


def job(model, seed, gpu, a):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(os.path.join(a.runs, f"log_{model}_s{seed}.txt"), "a")
    base = [sys.executable, "run_mri3d.py"]
    args = ["--model", model, "--seed", str(seed), "--data", a.data, "--out", a.runs, "--workers", "2"]
    cmd = (f"{' '.join(base)} train {' '.join(args)} && {' '.join(base)} test {' '.join(args)} "
           f"&& touch {a.runs}/{model}_s{seed}/test_done")
    return subprocess.Popen(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT)


def push(a, model):
    """Copy small result files into the repo and open/merge a results PR."""
    subprocess.run(f"mkdir -p results/mri3d && rsync -a --include='*/' --include='test.json' "
                   f"--include='train.json' --include='slices/*.npz' --exclude='*' {a.runs}/ results/mri3d/",
                   shell=True, check=False)
    stamp = time.strftime("%Y%m%d-%H%M")
    subprocess.run([sys.executable, "scripts/colab_pr.py", f"results/kaggle-{model}-{stamp}",
                    f"MRI results: {model} (Kaggle)", f"Test results for {model} on 98 held-out patients x 15 sequence patterns.",
                    "results/mri3d"], check=False)


def push_failure(a, model, seed):
    """Put the end of a failed model's log on GitHub so the error can be read without Kaggle access."""
    os.makedirs("results/mri3d/failures", exist_ok=True)
    log = os.path.join(a.runs, f"log_{model}_s{seed}.txt")
    tail = open(log).read().splitlines()[-80:] if os.path.exists(log) else ["(no log found)"]
    open(f"results/mri3d/failures/{model}_s{seed}.txt", "w").write("\n".join(tail) + "\n")
    stamp = time.strftime("%Y%m%d-%H%M")
    subprocess.run([sys.executable, "scripts/colab_pr.py", f"results/kaggle-failure-{model}-{stamp}",
                    f"MRI run failed: {model} (log tail)", "Last 80 lines of the training/test log, pushed automatically.",
                    "results/mri3d/failures"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/kaggle/working/runs")
    ap.add_argument("--data", default="/tmp/mri3d")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.runs, exist_ok=True)
    todo = [m for m in QUEUE if not is_done(a.runs, m, a.seed)]
    gpus = list(range(max(1, torch.cuda.device_count())))
    running = {}
    print(f"{len(todo)} models to run on GPUs {gpus}: {todo}", flush=True)
    while todo or running:
        for gpu in gpus:
            if gpu not in running and todo:
                m = todo.pop(0)
                running[gpu] = (m, job(m, a.seed, gpu, a))
                print(f"[{time.strftime('%H:%M')}] GPU {gpu}: start {m}", flush=True)
        time.sleep(30)
        for gpu, (m, p) in list(running.items()):
            if p.poll() is not None:
                print(f"[{time.strftime('%H:%M')}] GPU {gpu}: {m} finished (exit {p.returncode})", flush=True)
                del running[gpu]
                if p.returncode == 0:
                    push(a, m)
                else:
                    push_failure(a, m, a.seed)
    print("queue complete", flush=True)


if __name__ == "__main__":
    main()

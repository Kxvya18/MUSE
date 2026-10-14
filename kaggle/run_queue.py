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
    args = ["--model", model, "--seed", str(seed), "--data", a.data, "--out", a.runs, "--workers", "2"] + a.extra.split()
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


def report_progress(a, running, todo, finished):
    """Print each running model's latest log line and publish a progress file on the
    `kaggle-progress` branch (status only, never merged), so progress is visible from GitHub."""
    lines = [f"# MRI run progress ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})", ""]
    for gpu, (m, _) in running.items():
        log = os.path.join(a.runs, f"log_{m}_s{a.seed}.txt")
        last = [l for l in open(log).read().splitlines() if " it " in l or "val" in l or "test" in l][-1:] if os.path.exists(log) else []
        lines.append(f"- GPU {gpu}: **{m}** — {last[0].strip() if last else 'starting'}")
    lines += [f"- finished: {', '.join(finished) or '—'}", f"- waiting: {', '.join(todo) or '—'}"]
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    if os.environ.get("GH_TOKEN"):
        try:
            publish_progress(text)
        except Exception as e:  # progress reporting must never stop training
            print("progress publish failed:", e, flush=True)


def publish_progress(text, branch="kaggle-progress", path="PROGRESS.md"):
    """Write PROGRESS.md on a status branch via the GitHub API (no local git involved)."""
    import base64, json, urllib.request
    def api(method, url, payload=None):
        req = urllib.request.Request("https://api.github.com/repos/Kxvya18/MUSE" + url, method=method,
                                     data=json.dumps(payload).encode() if payload else None,
                                     headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                                              "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or b"{}")
    try:
        api("GET", f"/branches/{branch}")
    except urllib.error.HTTPError:
        sha = api("GET", "/git/ref/heads/main")["object"]["sha"]
        api("POST", "/git/refs", {"ref": f"refs/heads/{branch}", "sha": sha})
    try:
        old = api("GET", f"/contents/{path}?ref={branch}")["sha"]
    except urllib.error.HTTPError:
        old = None
    body = {"message": "progress", "branch": branch, "content": base64.b64encode(text.encode()).decode()}
    if old:
        body["sha"] = old
    api("PUT", f"/contents/{path}", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/kaggle/working/runs")
    ap.add_argument("--data", default="/tmp/mri3d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default="", help="extra arguments passed to run_mri3d.py train/test")
    ap.add_argument("--models", default=",".join(QUEUE), help="comma-separated subset of the queue to run")
    ap.add_argument("--progress_every", type=int, default=300, help="seconds between progress reports")
    a = ap.parse_args()
    os.makedirs(a.runs, exist_ok=True)
    todo = [m for m in a.models.split(",") if not is_done(a.runs, m, a.seed)]
    finished, last_report = [], 0.0
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
                    finished.append(m)
                    push(a, m)
                else:
                    push_failure(a, m, a.seed)
        if time.time() - last_report > a.progress_every:
            report_progress(a, running, todo, finished)
            last_report = time.time()
    report_progress(a, running, todo, finished)
    print("queue complete", flush=True)


if __name__ == "__main__":
    main()

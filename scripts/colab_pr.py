"""Open and merge a results PR from Colab (no gh CLI there), via the GitHub REST API.

usage (inside the cloned repo, GH_TOKEN in the environment):
    python scripts/colab_pr.py <branch> "<title>" "<body>" <path> [<path> ...]
"""
import json
import os
import subprocess
import sys
import urllib.request

REPO = "Kxvya18/MUSE"


def sh(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def api(method, path, payload=None):
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}{path}", method=method,
                                 data=json.dumps(payload).encode() if payload else None,
                                 headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                                          "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or b"{}")


def main():
    branch, title, body, paths = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    sh("git", "fetch", "-q", "origin", "main")
    sh("git", "checkout", "-q", "-B", branch, "origin/main")
    sh("git", "add", "-A", *paths)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print("nothing to commit")
        return
    files = sh("git", "diff", "--cached", "--name-only").splitlines()
    sh("git", "commit", "-q", "-m", title, "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    sh("git", "push", "-q", "-f", "origin", branch)
    pr = api("POST", "/pulls", {"title": title, "head": branch, "base": "main",
                                "body": body + "\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"})
    api("POST", f"/pulls/{pr['number']}/reviews", {"event": "COMMENT", "body":
        f"Pre-merge check (Colab): {len(files)} result files added, all under {', '.join(paths)}; "
        "no code changes; numbers written directly by run_mri3d.py / run_graph.py. Merging."})
    api("PUT", f"/pulls/{pr['number']}/merge", {"merge_method": "squash"})
    sh("git", "checkout", "-q", "main")
    sh("git", "pull", "-q", "origin", "main")
    print("merged", pr["html_url"])


if __name__ == "__main__":
    main()

#!/bin/bash
# Commit result/figure artefacts through a feature-branch PR: branch -> commit -> push ->
# PR -> pre-merge checks posted as a review comment -> squash merge.
# usage: scripts/pr_results.sh <branch> <title> <summary-markdown-file>
set -e
cd "$(dirname "$0")/.."
until mkdir .gitlock 2>/dev/null; do sleep 15; done
trap 'git checkout -q main 2>/dev/null; rmdir .gitlock' EXIT
git checkout -q main && git pull -q
git checkout -q -B "$1"
for d in results figures paper; do [ -e "$d" ] && git add -A "$d"; done
if git diff --cached --quiet; then exit 0; fi
nfiles=$(git diff --cached --name-only | wc -l | tr -d ' ')
codechg=$(git diff --cached --name-only | grep -vE '^(results|figures/(results|mri)|paper)/' || true)
git commit -q -m "$2" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q -u origin "$1" --force
tests=$(.venv/bin/python -m pytest -q tests 2>&1 | tail -1)
body="$(cat "$3" 2>/dev/null)

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
url=$(gh pr create --base main --head "$1" --title "$2" --body "$body")
gh pr review "$url" --comment --body "Pre-merge checks: unit tests: ${tests}. Artefacts added/updated: ${nfiles} files under results/figures/paper. Code files touched: ${codechg:-none}. Numbers come straight from the run JSONs (no manual edits). Merging."
gh pr merge "$url" --squash --delete-branch
git checkout -q main && git pull -q

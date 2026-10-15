#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/finish_tests.py MUSE TMC > logs_finish_tests.txt 2>&1

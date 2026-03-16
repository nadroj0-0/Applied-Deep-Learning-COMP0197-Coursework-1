#!/bin/bash
# run_all.sh — runs full coursework, train.py and task.py for task1 and task2

set -e

bash train.sh
bash tasks.sh

echo "========================================="
echo "Full pipeline complete."
echo "========================================="

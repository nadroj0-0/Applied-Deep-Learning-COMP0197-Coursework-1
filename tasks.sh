#!/bin/bash
# tasks.sh — runs task.py for task1 and task2

set -e

echo "========================================="
echo "Starting Task 1 evaluation..."
echo "========================================="
python task1/task.py

echo "========================================="
echo "Starting Task 2 evaluation..."
echo "========================================="
python task2/task.py

echo "========================================="
echo "All evaluation complete."
echo "========================================="

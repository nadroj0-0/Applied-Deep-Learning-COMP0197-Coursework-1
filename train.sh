#!/bin/bash
# train.sh — trains all models for task1 and task2

set -e  # exit immediately if any command fails

echo "========================================="
echo "Starting Task 1 training..."
echo "========================================="
python task1/train.py

echo "========================================="
echo "Starting Task 2 training..."
echo "========================================="
python task2/train.py

echo "========================================="
echo "All training complete."
echo "========================================="

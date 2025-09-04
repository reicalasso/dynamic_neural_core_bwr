#!/bin/bash
# train_small.sh

# Activate virtual environment
# source .venv/bin/activate # For Linux/macOS
# .\.venv\Scripts\Activate.ps1 # For Windows PowerShell

# Run training with the 'small' configuration
python -m bwr.trainer --config ../configs/small.yaml --run-name bwr_small_run

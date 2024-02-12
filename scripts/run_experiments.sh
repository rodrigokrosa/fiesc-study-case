#!/bin/bash

# Schedule execution of many runs
# Run from root folder with: bash scripts/run_experiments.sh

python src/train.py -m '+experiment/baseline=glob(*)'

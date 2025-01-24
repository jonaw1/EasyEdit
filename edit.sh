#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=250GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2

# python3 -m venv .venv
source .venv/bin/activate

# pip install -r requirements.txt

python3 my_edit.py

deactivate

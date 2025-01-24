#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --output=example.out
#SBATCH --error=error.out
#SBATCH --mem=80GB
#SBATCH --gpus=1

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python3 my_edit.py

conda deactivate

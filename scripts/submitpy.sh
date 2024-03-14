#!/bin/bash
#SBATCH --job-name=Permute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mfeigeli@ucsd.edu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

source /sphere/greene-lab/miniconda3/bin/activate greenelab

python3 step1_run_permute.py
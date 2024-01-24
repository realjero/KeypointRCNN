#!/bin/bash

#SBATCH --container-image ./container.sqfs
#SBATCH --gpus 1
#SBATCH --cpus-per-task 1
#SBATCH --time 2-23:59:59
#SBATCH --mem-per-gpu=26000MB
#SBATCH --output 2_run.out

#SBATCH --container-mounts=./:/workspace

python src/main.py
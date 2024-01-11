#!/bin/bash

#SBATCH --container-image ./container.sqfs
#SBATCH --gpus 1
#SBATCH --cpus-per-task 1
#SBATCH --time 2-00:00:00
#SBATCH --output 2_use.out

#SBATCH --container-mounts=./:/workspace

python main.py
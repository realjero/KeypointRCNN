#!/bin/bash

#SBATCH --container-image 'nvcr.io/nvidia/pytorch:23.12-py3'
#SBATCH --container-save 'container.sqfs'
#SBATCH --container-writable
#SBATCH --container-remap-root
#SBATCH --gpus 1
#SBATCH --cpus-per-task 1
#SBATCH --time 0-00:20:00
#SBATCH --output 1_prepare.out

pip install xtcocotools
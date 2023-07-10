#!/bin/bash
#SBATCH --job-name=jsort
#SBATCH --mail-type=ALL
#SBATCH --mail-user=afshin.khodaveisi@unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./sort.log
#SBATCH --error=./sort.err
#SBATCH --gres=gpu:1

nvcc ./sort_vec.cu -o sort_vec

./sort_vec


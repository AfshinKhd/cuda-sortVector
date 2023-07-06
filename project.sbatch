#### START SLURM CONFIGURATION ####

#!/bin/bash
#SBATCH --job-name=jsort
#SBATCH --mail-type=ALL
#SBATCH --mail-user=afshin.khodaveisi@unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./sort.log
#SBATCH --gres=gpu:1

#### END SLURM CONFIGURATION ####

#### START USER SCRIPT ####

####source ../venv/bin/activate # activates the Python virtual environment, installed under 'venv' folder

####Compile with:
nvcc ./src/sort_vec.cu -o cuda-sortvec

####Run with:
./src/sort_vec.cu

#### END USER SCRIPT ####
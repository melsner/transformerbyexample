#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --mem=24000
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=<ROOT>/<LOC>/%x.out
#SBATCH --error=<ROOT>/<LOC>/%x.err

NAME=<NAME>

echo "Job start $NAME"
echo "Working directory $(pwd)"

module load python/3.7-conda4.5
module load cuda
cd $SLURM_SUBMIT_DIR

echo "Changed to $(pwd)"

source activate tfgpu

echo "Path $PATH"
echo "Which python? $(which python)"
export PATH="/home/elsner.14/envs/tfgpu/bin:$PATH"
echo "Path $PATH"
echo "Which python? $(which python)"

python -u script/detectgpu.py

TF_DISABLE_MKL=1 python -u script/load_eval.py --run <ROOT>/<LOC>/$NAME --load_other <PARENT> --devset <DEV>

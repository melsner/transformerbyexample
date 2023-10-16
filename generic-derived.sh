#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --mem=24000
#SBATCH --time=20:00:00
#SBATCH --output=runs/derived/%x.out
#SBATCH --error=runs/derived/%x.err

NAME=<NAME>

echo "Job start $NAME"
echo "Working directory $(pwd)"

module load python/3.7-conda4.5
cd $SLURM_SUBMIT_DIR

echo "Changed to $(pwd)"

source activate tf2

echo "Path $PATH"
echo "Which python? $(which python)"
export PATH="/home/elsner.14/envs/tf2/bin:$PATH"
echo "Path $PATH"
echo "Which python? $(which python)"

TF_DISABLE_MKL=1 python -u script/byexample.py --run runs/derived/$NAME --data <TRAIN> --devset <DEV> --s2s_max_num_epochs=50 --s2s_patience=100 --load_other <PARENT> --allow_self_exemplar

#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --mem=24000
#SBATCH --time=24:00:00
#SBATCH --output=runs/parent/%x.out
#SBATCH --error=runs/parent/%x.err

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

TF_DISABLE_MKL=1 python -u script/byexample.py --run runs/parent/$NAME --data <TRAIN> --s2s_max_num_epochs=20 --s2s_patience=30 --load_other inflect_ubm_ml/model0/checkpoints/ckpt-30.index --append_train --edit_class=approximate

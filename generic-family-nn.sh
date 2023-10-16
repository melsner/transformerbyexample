#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --mem=24000
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=<ROOT>/fam/%x.out
#SBATCH --error=<ROOT>/fam/%x.err

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

TF_DISABLE_MKL=1 python -u script/byexample.py --run <ROOT>/fam/$NAME --data <TRAIN> --s2s_max_num_epochs=50 --s2s_patience=100 --load_other <PARENT> --allow_self_exemplar --exemplar_nn <EXNN> --n_exemplars=dynamic --limit_train=20000

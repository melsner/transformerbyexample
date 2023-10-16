#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --mem=48000
#SBATCH --time=18:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=<ROOT>/fam/%x.out
#SBATCH --error=<ROOT>/fam/%x.err
#SBATCH --account=PAS1957

NAME=<NAME>

echo "Job start $NAME"
echo "Working directory $(pwd)"

module load python/3.7-2019.10
module load cuda
cd $SLURM_SUBMIT_DIR

echo "Changed to $(pwd)"

source activate tfgpu

echo "Path $PATH"
echo "Which python? $(which python)"
export PATH="/users/PAS1268/osu8210/.conda/envs/tfgpu/bin:$PATH"
echo "Path $PATH"
echo "Which python? $(which python)"

python -u script/detectgpu.py

#2020: run 50 epochs
#2022: run 10
TF_DISABLE_MKL=1 python -u script/byexample.py --run <ROOT>/fam/$NAME --data <TRAIN> --s2s_max_num_epochs=10 --s2s_patience=100 --load_other <PARENT> --allow_self_exemplar --n_exemplars=dynamic --limit_train=20000 --extra_features=numeric

#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --mem=24000
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=<ROOT>/fine/%x.out
#SBATCH --error=<ROOT>/fine/%x.err
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

TF_DISABLE_MKL=1 python -u script/byexample.py --run <ROOT>/fine/$NAME --data <TRAIN> --devset <DEV> --s2s_max_num_epochs=0 --load_other <PARENT> 

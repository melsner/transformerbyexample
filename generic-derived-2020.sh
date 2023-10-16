#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --mem=48000
#SBATCH --time=18:00:00
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

TF_DISABLE_MKL=1 python -u script/byexample.py --run <ROOT>/fine/$NAME --data <TRAIN> --devset <DEV> --s2s_max_num_epochs=50 --s2s_patience=100 --load_other <PARENT> --allow_self_exemplar --extra_features=numeric

#eval for basic
#TF_DISABLE_MKL=1 python -u script/writeOutput.py <DEV> <ROOT>/fine/$NAME/model0/predictions_dev.txt

#eval block
echo "--------------- predicting ------------------"
TF_DISABLE_MKL=1 python -u script/load_eval.py --run <ROOT>/fine/$NAME --devset <ROOT>/fine/$NAME/dev.txt
echo "--------------- creating featurized file for second pass -------------"
TF_DISABLE_MKL=1 python -u script/usePredictedFeatures.py <ROOT>/fine/$NAME/$NAME-dev.txt/predictions_dev.txt --numeric
echo "--------------- predicting ------------------"
TF_DISABLE_MKL=1 python -u script/load_eval.py --run <ROOT>/fine/$NAME --devset <ROOT>/fine/$NAME/$NAME-dev.txt/predictions_dev-predicted.txt 
echo "---------- write output --------------"
TF_DISABLE_MKL=1 python -u script/writeOutput.py <DEV> <ROOT>/fine/$NAME/$NAME-dev.txt-predictions_dev-predicted.txt/predictions_dev.txt > <ROOT>/fine/$NAME/writeOutput.log

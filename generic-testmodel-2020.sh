#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --mem=24000
#SBATCH --time=02:00:00
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

RUN=<ROOT>/fine/$NAME

echo "--------------- creating file ------------------"
TF_DISABLE_MKL=1 python -u script/byexample.py --run $RUN --data <TRAIN> --devset <DEV> --generate_file --extra_features=all
echo "--------------- predicting ------------------"
F_DISABLE_MKL=1 python -u script/load_eval.py --run $RUN --devset $RUN/dev.txt --load_other <PARENT>
echo "--------------- creating featurized file for second pass ------------------"
TF_DISABLE_MKL=1 python -u script/usePredictedFeatures.py $RUN/$NAME-dev.txt/predictions_dev.txt
echo "--------------- predicting ------------------"
TF_DISABLE_MKL=1 python -u script/load_eval.py --run $RUN --devset $RUN/$NAME-dev.txt/predictions_dev-predicted.txt --load_other <PARENT>

echo "---------- write output --------------"
TF_DISABLE_MKL=1 python -u script/writeOutput.py <DEV> $RUN/$NAME-dev.txt-predictions_dev-predicted.txt/predictions_dev.txt > $RUN/output.log

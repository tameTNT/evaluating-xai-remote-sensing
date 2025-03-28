#!/bin/bash
# The first argument to this script should be the name of job
# The second argument should be the path to a file containing the arguments for evaluate_batch.py
# The third argument should the number of hours to run the job for (usually 05 is fine)
# Any remaining arguments are passed to evaluate_batch.py

# work way up the path to get the relevant parameters
model_name=$(basename "$2")

dataset_path=$(dirname "$2")
dataset_name=$(basename "$dataset_path")

explainer_path=$(dirname "$dataset_path")
explainer_name=$(basename "$explainer_path")

args=$(cat "$2")

printf '%s\n' "#!/bin/bash" \
       "#SBATCH -N 1" \
       "#SBATCH -c 4" \
       "#SBATCH -p ug-gpu-small" \
       '#SBATCH --qos="short"' \
       "#SBATCH --mem=28G" \
       "#SBATCH -t 00-$3:00:00" \
       "#SBATCH --gres=gpu:1g.10gb:1" \
       "#SBATCH --mail-user=jgcw74@durham.ac.uk" \
       "#SBATCH --mail-type=FAIL" \
       "#SBATCH --job-name=$1" \
       "#SBATCH -o /home2/jgcw74/l3_project/.logs/slurm/$1.log" \
       "##Activate conda and environment" \
       "eval $(conda shell.bash hook)" \
       "conda activate sat_project" \
       "cd ~/l3_project" \
       "export DATASET_ROOT=~/datasets" \
       "$(printf '%s ' "python batch_evaluate.py --dataset_name $dataset_name --model_name $model_name --explainer_name $explainer_name $args ${@:4}")" > ~/l3_project/slurm_ncc/evaluate_temp.batch

sbatch ~/l3_project/slurm_ncc/evaluate_temp.batch

#!/bin/bash
# The first argument to this script should be the name of run/(WandB) job
# The second argument should be a path to the .args file to use to run the job
# The remaining arguments are passed to run_train.sh
printf '%s\n' "#!/bin/bash" \
       "#SBATCH -N 1" \
       "#SBATCH -c 4" \
       "#SBATCH -p ug-gpu-small" \
       '#SBATCH --qos="short"' \
       "#SBATCH --mem=28G" \
       "#SBATCH -t 00-09:00:00" \
       "#SBATCH --gres=gpu:1g.10gb:1" \
       "#SBATCH --mail-user=jgcw74@durham.ac.uk" \
       "#SBATCH --mail-type=BEGIN" \
       "#SBATCH --job-name=$1" \
       "#SBATCH -o /home2/jgcw74/l3_project/.logs/slurm/$1.log" \
       "##Activate conda and environment" \
       "eval $(conda shell.bash hook)" \
       "conda activate sat_project" \
       "cd ~/l3_project" \
       "export DATASET_ROOT=~/datasets" \
       "$(printf '%s ' "bash run_train.sh $2 --wandb_run_name $1 ${@:3}")" > ~/l3_project/slurm_ncc/train_model_temp.batch

sbatch ~/l3_project/slurm_ncc/train_model_temp.batch

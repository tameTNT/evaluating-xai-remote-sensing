#!/bin/bash
# Execute in cwd with e.g. source run_train.sh model_training/config/help.args (Windows)
# or ./run_train.sh model_training/config/help.args (Unix)

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate sat_project

# Read provided file and notes
pattern="^\s*#|^\s*$"
echo "Notes:"
grep -E "$pattern" $1
echo

# Run python command
txt_args=$(grep -vE "$pattern" $1)

echo "Using following args from $1 to run train.py:"
echo $txt_args "${@:2}"
echo
python train.py $txt_args "${@:2}"

#!/bin/bash

# Activate conda environment
cd ~/l3_project/
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
echo $txt_args
python train.py $txt_args

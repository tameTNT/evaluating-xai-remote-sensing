#!/bin/bash
# Execute in cwd with e.g. source run_train.sh model_training/config/help.args (Windows)
# or ./run_train.sh model_training/config/help.args (Unix)

if [[ $(hostname | grep -E "(gpu|Luca)" | wc -c) -ne 0 ]]; then
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

  echo "Using following args (last value takes precedence if multiple) from $1 to run train.py:"
  echo $txt_args "${@:2}"
  echo
  python -m model_training.train $txt_args "${@:2}"
else
  echo "Not on a GPU! Exiting..."
  exit 1
fi

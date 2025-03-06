#!/bin/bash

if squeue -u jgcw74 --json | grep \"Ampsat_port_jupyter\" | head -c1 | grep -E '.'; then
  echo "Existing job with same name. Exiting."
  exit 1
fi
echo "No existing jobs with same name. Starting job"
sbatch ~/l3_project/slurm_ncc/jupyter_on_port_amp.batch

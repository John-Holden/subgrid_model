#!/bin/bash

# Execute me from terminal to run the sub-grid model, simulate me in multiple realisations thus forming an ensemble.
# Run: 'qsub run_sgm.sh' on leeds HPC to use a task array of n cores.
# either on the HPC or locally.

###########__________Run script__________#############
################ Hpc machine ################

module load python/3.6.5
module load python-libs/3.1.0
date_time=$(date '+%d-%m-%Y %H:%M:%S')

#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -t 1-100

mode="HPC"
sim_type="-2D-phase"
sim_name="-R0"

python3 mkdir.py $date_time $mode $sim_type $sim_name
python3 sg_main.py $SGE_TASK_ID $date_time $data_type $mode $sim_type $sim_name

echo "Simulations Finished"

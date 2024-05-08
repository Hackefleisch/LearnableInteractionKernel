#!/bin/bash -l

module load Anaconda3

source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate /beegfs/.global1/ws/paei790f-learnable_kernel/conda_env

# Load your script. $@ is all the parameters that are given to this run.sh file.
python ~/LearnableInteractionKernel/train.py $@
#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=48:00:00
#SBATCH --job-name=BioKernels
#SBATCH --array=0-5



interactions=( 'hydrophobic', 'hbond', 'pistacking', 'pication', 'saltbridges', 'halogenbond' )

srun ~/LearnableInteractionKernel/train_hpc.sh --epochs 1000 --num_workers 6 --abs_path /beegfs/ws/1/paei790f-learnable_kernel/ --data_dir pdbbind2020_reduced/ --split_file LP_PDBBind.csv --storage_path model_weights/ --interactions ${interactions[$SLURM_ARRAY_TASK_ID]} --save_weights --full_dataset_on_gpu

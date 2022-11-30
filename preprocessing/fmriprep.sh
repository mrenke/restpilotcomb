#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --output=/home/cluster/mrenke/logs/res_fmriprep_%A-%a.txt
#SBATCH --partition=generic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
export SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow
singularity run -B /scratch/$USER/templateflow:/opt/templateflow --cleanenv /data/$USER/containers/fmriprep-20.2.3.simg /scratch/$USER/data/ds-restpilotcomb /scratch/$USER/data/ds-restpilotcomb/derivatives participant --participant-label $PARTICIPANT_LABEL  --output-spaces MNI152NLin2009cAsym T1w fsaverage fsaverage5 fsnative  --dummy-scans 3 --skip_bids_validation -w /scratch/$USER/workflow_folders --no-submm-recon
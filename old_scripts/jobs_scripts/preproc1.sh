#!/bin/bash
# 
# Job name 
#$ -N preproc1
#$ -l h_rt=06:00:00
#
# Log file
#$ -o /home/tiziano.causin/output_logs/$JOB_NAME.$JOB_ID.log
# 
# Merge error and output
#$ -j yes 
# 
# RAM required
#$ -l h_vmem=100G
#$ -pe smp 6
###################
cd /mnt/storage/tier2/ingdev/projects/TIZIANO/Project1917/code/preprocessing/cluster_scripts/
# Call Matlab
/state/partition1/MATLAB/R2020a/bin/matlab -nodesktop -nosplash -nodisplay -r "cProject1917_preproc1_reref_filter_trim; exit;" 




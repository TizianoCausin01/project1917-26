#!/bin/bash
# 
# Job name 
#$ -N preproc5
#$ -l h_rt=20:00:00
#
# Log file
#$ -o /home/tiziano.causin/output_logs/$JOB_NAME.$JOB_ID.log
# 
# Merge error and output
#$ -j yes 
# 
# RAM required
#$ -l h_vmem=250G
#$ -pe smp 8
###################
cd /mnt/storage/tier2/ingdev/projects/TIZIANO/Project1917/code/preprocessing/cluster_scripts/
# Call Matlab
/state/partition1/MATLAB/R2020a/bin/matlab -nodesktop -nosplash -nodisplay -r "cProject1917_preproc5_4TizianoSISSA; exit;" 




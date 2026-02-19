#!/bin/bash
# 
# Job name 
#$ -N downsampling 
#$ -l h_rt=06:00:00
#
# Log file
#$ -o /home/tiziano.causin/output_logs/$JOB_NAME.$JOB_ID.log
# 
# Merge error and output
#$ -j yes 
# 
# RAM required
#$ -l h_vmem=20G
#
###################
cd $SGE_O_WORKDIR
# Call Matlab
/state/partition1/MATLAB/R2020a/bin/matlab -nodesktop -nosplash -nojvm -nodisplay -r "cProject1917_downsampling; exit;" 




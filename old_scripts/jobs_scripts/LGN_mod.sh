#!/bin/bash
# 
# Job name
#$ -N LGN_mod 
#$ -l h_rt=72:00:00
#
# Log file
#$ -o /home/tiziano.causin/output_logs/$JOB_NAME.$JOB_ID.log
# 
# Merge error and output
#$ -j yes 
# 
# RAM required
#$ -l h_vmem=250G



#    -pe smp 3

###################
cd /mnt/storage/tier2/ingdev/projects/TIZIANO/Project1917/code/dRSA/cluster_scripts/

# Call Matlab
/state/partition1/MATLAB/R2020a/bin/matlab -nodesktop -nosplash -nodisplay -r "compute_LGN_mod_cluster; exit;"



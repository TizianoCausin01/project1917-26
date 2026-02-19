#!/bin/bash
# 
# Job name 
#$ -N get_vid_stats
#
# Script interpreter
#$ -S /bin/bash  
#
# Log file
#$ -o /home/tiziano.causin/output_logs/$JOB_NAME.$JOB_ID.log
# 
# Merge error and output
#$ -j yes 

#defines the path to the script (otherwise you'll get redirected to the initial directory)
cd $SGE_O_WORKDIR
conda activate 1917_py_env
python3 /mnt/storage/tier2/ingdev/projects/TIZIANO/Project1917/code/py_scripts/get_vid_stats.py
exit



#!/bin/bash
# 
# Job name 
#$ -l h_rt=24:00:00
#
# Log file
#$ -o /home/tiziano.causin/output_logs/$JOB_NAME.$JOB_ID.log
# 
# Merge error and output
#$ -j yes 
# 
# RAM required
#$ -l h_vmem=250G
###################
cd /mnt/storage/tier2/ingdev/projects/TIZIANO/Project1917/code/dRSA/cluster_scripts/
sub=${1}
mod=${2} # in the form of "'mod1', 'mod2',..."
roi=${3} # either 1 or in the form "'occ', 'par', ..."
igaze=${4}
ires=${5}
nproc=${6}

echo sub $sub
echo mod $mod
echo roi $roi
echo igaze $igaze
echo ires $ires
echo nproc $nproc

# Call Matlab
/state/partition1/MATLAB/R2020a/bin/matlab -nodesktop -nosplash -nodisplay -r "cProject1917_dRSA_general(${sub}, {${mod}}, {${roi}},${igaze}, ${ires}, ${nproc}); exit;"



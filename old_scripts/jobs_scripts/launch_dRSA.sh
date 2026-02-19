#!/bin/bash

#subjects=${1} # like this "1 2 ..."
mod=${2} # in the form of "'mod1', 'mod2',..."
rois=${3} # either 1 or in the form "'occ', 'par', ..."
igaze=${4} # 1 if gaze dep, 0 if gaze indep
ires=${5} # 24 if movie res, 50 if eyetracking signal res
nproc=${6}
echo mod $mod
echo rois $rois
echo nproc $nproc
echo ires $ires
echo igaze $igaze
read -a subjects <<< "$1"    # reads the first argin and creates an array called files
for sub in "${subjects[@]}"; do
    qsub -N dRSA_sub${sub} -pe smp ${nproc} dRSA_general.sh ${sub} "${mod}" "${rois}" ${igaze} ${ires} ${nproc} 
done

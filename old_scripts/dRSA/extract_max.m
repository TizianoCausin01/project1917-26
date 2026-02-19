% "alexnet_fc_layer5"
%
results_dir = "/Volumes/TIZIANO/results/corr/50Hz_180stim_10sec_100iter_0MNN";
parms=[]
parms.subjects = [3:10];
parms.fsNew = 50;
upper_y_lim = 0.006;
rois = ["occ", "par", "tem", "fro"]
%mods = ["CE", "gamma", "beta"]
% mods = ["gbvs_map", "gbvs_map_KLD", "dg_map", "dg_map_KLD"]
mods = ["ViTPose"] %, "dg_map"]
imod_tit = ["ViTPose" ] %, "Deep-Gaze II"]

roi_counter = 0
for iroi = rois
    roi_counter = roi_counter+1;
    counter = 0
    for imod = mods
        counter = counter+1;
        sm{roi_counter} = peak_latency(results_dir, imod,iroi, parms)
    end
end

function pl = peak_latency(results_directory,imod,iroi,parms)
pl = {zeros(size(parms.subjects,2),2)}
freq = round(parms.fsNew);
for irep=[1 2]
    count=0;
    isub_counter = 0;
    for isub=parms.subjects
        count=count+1;
        isub_counter = isub_counter +1
        fn2load = sprintf('%s/sub%03d/dRSA_corr_sub%03d_%s_%s_rep%d_%dHz.mat', results_directory, isub, isub,imod,iroi,irep,freq);
        disp(fn2load)
        load(fn2load)
        [t, pl(isub_counter,irep)] = max(dRSA, [], 2);
    end
end
end %EOF

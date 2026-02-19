clear all
models_dir = "/Volumes/TIZIANO/models";
dataDir = "/Volumes/TIZIANO/eyetracking_data";
runs = 1:3;
reps = 1:2;
subjects = 3:10;
resHor = 1920;% horizontal screen resolution
resVer = 1080;% vertical screen resolution
height = 522;
width = 1280;
fsVid = 23.976;
parms.fsNew = 50;
bufferVer = (resVer-height)/2;
bufferHor = (resHor-width)/2;
frames2discard = 119;

for irun = runs
    fprintf("%s starting run %d \n", string(datetime('now')), irun)
    % load and resample movie mod
    mod2load = sprintf("%s/Project1917_ViTPose_run%02d.h5", models_dir, irun);
    body_posture_mod = h5read(mod2load, "/kpts");
    body_posture_mod(:,1:frames2discard) = []; % discards the first five secs (119 pts) of each run
    tVid = 0:1/fsVid:size(body_posture_mod,2)/fsVid-1/fsVid;
    tNew = 0:1/parms.fsNew:size(body_posture_mod,2)/fsVid-1/parms.fsNew;
    % for each new time point, find index of nearest old time point
    old2newID = dsearchn(tVid',tNew');
    body_posture_mod = body_posture_mod(:,old2newID,:);
    tVid = tNew; % change name for better compatibility
    vecrep = []; % resets the size of vecrep across runs 
    for isub = subjects
    fprintf("%s starting sub %d of run %d \n", string(datetime('now')), isub, irun)    
        for irep = reps
            fprintf("%s starting rep %d of sub %d of run %d \n", string(datetime('now')), irep, isub, irun)    
            % load gaze data
            if irep==1
                fn2load=sprintf('%s/gaze_sub%03d_run%02d_50Hz.mat',dataDir,isub,irun);
            else
                fn2load=sprintf('%s/gaze_sub%03d_run%02d_50Hz.mat',dataDir,isub,irun+3);
            end % if irep==2
            load(fn2load);
            gaze_data = gaze([1 2],:);

            for itime = 1:length(body_posture_mod)
                
                % select body-posture
                x = gaze_data(1,itime) - bufferVer;
                y = gaze_data(2,itime) - bufferHor;
                [kpts_arr, idx] = get_closest_kpts(body_posture_mod(:,itime), x, y, "min_dist");
                if all(isnan(kpts_arr(:))) == 1
                    %nothing happens (keeps kpts_vec like the previous one)
                    fprintf("%s interpolating frame %d of rep %d of sub %d of run %d that was nan \n", string(datetime('now')), itime, irep, isub, irun)              
                else
                    kpts_vec = reshape(kpts_arr, [], 1);
                end % if isnan(kpts) == 1

                vecrep(:, itime) = kpts_vec;
                % fprintf("%s computed frame %d of rep %d of sub %d of part
                % %d \n", string(datetime('now')), itime, irep, isub, irun)
                % % commented because it's too fast
            end % for itime = 1:length(vecrep_new)

            if length(tVid) ~= length(vecrep)
                error("the size of the model doesn't match the size of the timepts")
            end % if length(tVid) ~= length(vecrep)

            % save
            if irep == 1
                fn2save =sprintf("%s/gaze_dep_mod/Project1917_ViTPose_sub%03d_run%02d_movie%dHz.mat", models_dir, isub, irun, parms.fsNew);
            else
                fn2save =sprintf("%s/gaze_dep_mod/Project1917_ViTPose_sub%03d_run%02d_movie%dHz.mat", models_dir, isub, irun+3, parms.fsNew);
            end % if irep == 1
            save(fn2save, 'vecrep', 'fsVid', 'tVid')
        end %for irep = reps
    end % for isub = subjects
end % for irun = runs
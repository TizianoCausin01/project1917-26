function Project1917_dRSA_core(preproc_dir, models_dir, results_dir,  parms,isub,irep,imod,iroi,ires, igaze, roi_name, mod_name)
% input and output folders
indirMEG = sprintf('%s%ssub-%03d%spreprocessing', preproc_dir,filesep,isub,filesep);
if igaze == 1% gaze-dependent models
    indirMOD = sprintf('%s/gaze_dep_mod',models_dir); 
else% gaze-invariant models
    % indirMOD = sprintf('%s%smodels',models_dir,filesep);
    indirMOD = models_dir;
end

if parms.similarity == 0
    simstring = 'corr';
elseif parms.similarity == 1
    simstring = ['pcr_' num2str(parms.nPCRcomps) 'comps'];
end

outdir = sprintf('%s%s%s%s%dHz_%dstim_%dsec_%diter_%dMNN%ssub%03d',results_dir,filesep,simstring,filesep,parms.fsNew,parms.nstim,parms.stimlen,parms.iterations,parms.MNN,filesep,isub);
if ~exist(outdir,'dir')
    mkdir(outdir);
end

%!!CHANGED not sure what it does

% load MEG and model data
% MEG data
% fn2load = sprintf('%s%ssub%03d_%s_%dHz_MNN%d_badmuscle%d_badlowfreq%d_badsegint%d_badcomp%d',...
%     indirMEG,filesep,isub,parms.ROInames{iroi},parms.fsNew,parms.MNN,parms.rej_bad_muscle,parms.rej_bad_lowfreq,parms.bad_seg_interp,parms.rej_bad_comp);
fn2load = sprintf('%s%ssub%03d_%s_%dHz',...
    indirMEG,filesep,isub,parms.ROInames{iroi},parms.fsNew);
disp(fn2load)
% load(fn2load,'data_final'); #FIXME for rep1_2 load both repetitions and
% preproc both
load(fn2load)
data_final = data.trial; % my preproc5 didn't have data_final, so I am doing it now
% select current repetition
runIDs = 3*irep-2:3*irep;
data_final = data_final(runIDs);

% if a sensor was bad and is NaN in one movie part, it should be removed from all three
badchan = false(2,size(data_final{1},1));
for ipart = 1:3
    badchan(ipart,:) = all(isnan(data_final{ipart}),2) | all(data_final{ipart} == 0,2);
end

badchan = logical(sum(badchan));

% load model data, and cut MEG data to size
vecrep_all = cell(length(runIDs),1);
count = 0;
for irun = runIDs
    if ~strcmp(imod, "rep1_2")
        count = count + 1;
        if igaze == 1% gaze dependent models  %FIXME add correct folder
            fn2load = sprintf('%s/Project1917_%s_sub%03d_run%02d_movie%dHz',indirMOD,parms.modelnames{imod},isub,irun,ires);
        else% gaze invariant models
            fn2load = sprintf('%s%sProject1917_%s_run%02d_movie%dHz',indirMOD,filesep,parms.modelnames{imod},count,ires);
        end
        load(fn2load,'vecrep','fsVid','tVid');
        if ires ~= 50
            % resample to new sampling rate in parms.fsNew, only 'nearest' makes sense for movie frames
            tNew = 0:1/parms.fsNew:size(vecrep,2)/fsVid-1/parms.fsNew; %FIXME put an if statement if fsVid ~= parms.fsNew

            % for each new time point, find index of nearest old time point
            old2newID = dsearchn(tVid',tNew');
            vecrep = vecrep(:,old2newID,:);
        end % if ires ~= 50
        % cut MEG data at end because we stored 1 sec after the trigger signaling the end of the movie
        data_final{count}(:,size(vecrep,2)+1:end) = [];
        vecrep_all{count} = vecrep;
        clear vecrep
    elseif strcmp(imod, "rep1_2") % comparison rep 1 vs rep 2
        if irep ==1
            error("When you do rep 1 vs rep 2 you can't select rep 1")
        end
        count = count + 1;
        fn2OFdir = sprintf('%s%sProject1917_OFdir_run%02d_movie%dHz',indirMOD,filesep,count,ires);
        load(fn2OFdir,'tVid');

        % resample to new sampling rate in parms.fsNew, only 'nearest' makes sense for movie frames
        tNew = 0:1/parms.fsNew:size(vecrep,2)/fsVid-1/parms.fsNew;

        % for each new time point, find index of nearest old time point
        old2newID = dsearchn(tVid',tNew');
        vecrep = vecrep(:,old2newID,:);

        % cut MEG data at end because we stored 1 sec after the trigger signaling the end of the movie
        data_final{count}(:,size(vecrep,2)+1:end) = [];

        vecrep_all{count} = vecrep;
        clear vecrep
    end
end

% There is still ~3 sec overlap remaining between first and second movie part, but exact sample number might be slightly different due to several resampling steps
matchID = 152;% just set to 152 because determined with view-invariant pixelwise model, which should then hold for all

% now cut the overlap
vecrep_all{2}(:,1:matchID) = [];
data_final{2}(:,1:matchID) = [];

% concatenate the three movie parts
dataMEG = cat(2,data_final{1},data_final{2},data_final{3});
dataMEG(badchan,:) = [];
clear data_final

% remember last sample of second movie part for later mask
part2endID = size(vecrep_all{1},2)+size(vecrep_all{2},2);

% now concatenate the models
dataMOD = cat(2,vecrep_all{1},vecrep_all{2},vecrep_all{3});
clear vecrep_all


% create indices for random subsampling of cfg.nstim pseudo-stimuli over cfg.iterations iterations
% make sure rand numbers are different for each subject, each ROI, and each time this script is ran at a different time and date
rng(isub*10^7+iroi*10^6+second(datetime('now'))*10^4);
framenumtot = size(dataMEG,2);

% create mask with samples to ignore for random subsampling of pseudo-stimuli, use middle of movie parts 2 and 3 and any NaNs in the data
mask = false(1,framenumtot);
mask(part2endID-parms.stimlen*parms.fsNew+2:part2endID) = true;

% find any NaNs in MEG data and count backwards to ignore all onset indices
% that would cause a pseudo-trial to overlap with the NaNs, uint16 operation changes NaNs to zeros
badsegs = find(isnan(dataMEG(1,:)));
mask(badsegs-parms.stimlen*parms.fsNew+2:badsegs) = true;

% check if requested pseudo-trials fit in data length and otherwise give error
if parms.nstim*parms.stimlen*parms.fsNew + (parms.nstim-1)*parms.minISI*parms.fsNew > framenumtot
    error(['The requested combination of ' num2str(parms.nstim) ' pseudo-trials of ' num2str(parms.stimlen) ' sec and a minimum ISI of ' num2str(parms.minISI) ' sec does not fit in the data'])
end

% cumulative sum of pseudo-stimulus length
stimlencumsum = cumsum(parms.stimlen*parms.fsNew*ones(parms.nstim-1,1));

% cumulative sum of minimum inter-stimulus-interval
minISIcumsum = cumsum(parms.minISI*parms.fsNew*ones(parms.nstim-1,1));

% total length in frames of inter-stimulus-interval to divide in (nstim-1) parts
ISItotal = framenumtot - parms.stimlen*parms.nstim*parms.fsNew - minISIcumsum(end);

% initialize onsetIDs = start times [iterations X stimuli];
onsetIDiter = zeros(parms.iterations,parms.nstim);

% loop over iterations to create onsetIDs
for iter = 1:parms.iterations

    constraint = 1;% create new onsetIDs for this iteration until non overlap anywhere with the mask
    while constraint

        onsetID = rand(parms.nstim+1,1);% create vector of onsetIDs
        onsetID = onsetID/sum(onsetID)*ISItotal;% normalize to ISItotal so ISItotal gets divided in random parts
        onsetID(end) = [];% last one can be removed because it's after last pseudo-stimulus
        onsetID = cumsum(onsetID);% cumulative sum
        onsetID(2:end) = onsetID(2:end) + minISIcumsum;% add minimum inter-stimulus-interval between stimuli, skip interval before first stimulus
        onsetID(2:end) = onsetID(2:end) + stimlencumsum;% add stimulus length between stimuli, skip first onset
        onsetID = round(onsetID);

        % in rare instances, onsetID could now be 0, so just move to 1
        onsetID(onsetID==0) = 1;

        % only if no values in onsetID overlap with our mask, can we move
        % to the next iteration
        if sum(sum(onsetID == find(mask))) == 0
            constraint = 0;
        end

    end

    % once a good onsetID vector has been found, it can be added:
    onsetIDiter(iter,:) = onsetID;

end


% compute dRSA across iterations
% set some parameters
framenum = parms.stimlen*parms.fsNew;
tRange=parms.maxlatency*parms.fsNew;% in samples
latencytime = -parms.maxlatency:1/parms.fsNew:parms.maxlatency;
pseudotime = 0:1/parms.fsNew:framenum/parms.fsNew-1/parms.fsNew;

dRSAlatency = zeros(parms.iterations,length(latencytime),'single');
for iter = 1:parms.iterations

    clc;
%    disp([string(datetime('now')) ' Running dRSA iteration ' num2str(iter) ' of ' num2str(parms.iterations)]);
    fprintf("%s running dRSA iteration %d of %d", string(datetime('now')), iter, parms.iterations)

    % loop over frames of pseudo-stimuli and compute RDM at each frame
    neuralRDM = zeros(parms.nstim,parms.nstim,framenum,'single');
    modelRDM = neuralRDM;
    for iframe = 1:framenum

        % select current frame for all pseudo-stimuli
        frameIDs = onsetIDiter(iter,:)+iframe-1;

        dataMEGiter = single(dataMEG(:,frameIDs));
        dataMODiter = single(dataMOD(:,frameIDs));

        % center data across stimuli before computing RDMs (recommended by cosmoMVPA)
        %         dataMEGiter = dataMEGiter - mean(dataMEGiter,2);
        %         dataMODiter = dataMODiter - mean(dataMODiter,2);

        % compute RDM for current frame
        neuralRDM(:,:,iframe) = 1 - corr(dataMEGiter);
        modelRDM(:,:,iframe) = 1 - corr(dataMODiter); % FIXME add corr w/ nans (not needed yet, since we are interpolating)

    end% frame loop

    % take lower triangle from square RDM and vectorize
    neuralRDMvec = zeros((parms.nstim*parms.nstim-parms.nstim)/2,framenum,'single');
    modelRDMvec = neuralRDMvec;
    for iframe = 1:framenum
        neuralRDMvec(:,iframe) = squareform(tril(squeeze(neuralRDM(:,:,iframe)),-1));
        modelRDMvec(:,iframe) = squareform(tril(squeeze(modelRDM(:,:,iframe)),-1));
    end

    % temporally smooth RDMs
    if parms.smoothNeuralRDM
        neuralRDMvec = ft_preproc_smooth(neuralRDMvec,parms.smoothNeuralRDM);
    end
    if parms.smoothModelRDM
        modelRDMvec = ft_preproc_smooth(modelRDMvec,parms.smoothModelRDM);
    end

    % run dRSA
    if parms.similarity == 0% simple correlation

        % neural - model correlation
        dRSAiter = corr(modelRDMvec,neuralRDMvec);

    end

    % slice neural vectors per model time point, and then stack, to create neural - model latency
    dRSAstack = zeros(length(pseudotime),length(latencytime));
    for iModelTime = 1:length(pseudotime)

        timeidx = iModelTime - tRange:iModelTime + tRange;
        NotInVid = logical((timeidx < 1)+(timeidx > length(pseudotime)));
        timeidx(NotInVid) = 1;%remove indices that fall before or after video

        slice = squeeze(dRSAiter(iModelTime,timeidx));
        slice(NotInVid) = NaN;%remove indices that fall before or after video
        dRSAstack(iModelTime,:) = slice;

    end

    %Average over video time
    dRSAlatency(iter,:) = squeeze(mean(dRSAstack,1,'omitnan'));

end

% average over iterations
dRSA = squeeze(mean(dRSAlatency));
fn2save = sprintf("%s%sdRSA_%s_sub%03d_%s_%s_rep%d_%dHz.mat", outdir, filesep, simstring, isub, mod_name, roi_name, irep, parms.fsNew);
% save dRSA results
save(fn2save,'dRSA','latencytime');
disp("done:")
disp(fn2save)
end %EOF


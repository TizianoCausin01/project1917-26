close all; clearvars; clc
numCores = 3; % to run in parallel the 6 runs
parpool('local',numCores)
addpath('/Users/tizianocausin/Desktop/programs/fieldtrip-20240110')
% set directories
% rootdir = '\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\Project1917';
preproc_dir = '/Volumes/TIZIANO/data_preproc';
models_dir = "/Volumes/TIZIANO/models"; 
results_dir = "/Volumes/TIZIANO/results";
% addpath(genpath(rootdir));
% codeDir='/Users/tizianocausin/Desktop/backUp20240609/summer2024/dondersInternship/code'; %!!CHANGED TO ADD THE PATH TO THE CODE WHICH IS SEPARATE FROM THE DATA REP
% addpath(genpath(codeDir))
% set dRSA parameters
% general parameters
parms = [];
parms.fsNew = 50;% here match the chosen neural sampling rate
% parms.subjects = 3:15;
parms.subjects = 3:4; %!!CHANGED BC I ONLY HAVE SUB03
parms.repetitions = 1:2;
parms.ROInames = {'allsens', 'occpar','occ', 'par', 'tem', 'fro'};
parms.ROI = 1:6;% 1 = all MEG sensors, 2 = occipito-parietal
parms.modelnames = {'pixelwise'} %,'OFmag','OFdir'};
parms.models2test = 1:3;% 
parms.OFtempres = 24;% whether to use OF models computed on 12 Hz downsampled or 24 Hz original movie data, or on both
parms.gazedep = 0;% whether to use  gaze-invariant models (0), or gaze-dependent models (1) or both

% preprocessing parameters
parms.gazeradius = 250;% circle size around gaze location in pixels
parms.rej_bad_muscle = 0;% 0 to keep bad muscle segments in, 1 to reject them
parms.rej_bad_lowfreq = 1;% 0 to keep low-freq noise segments in, 1 to reject them
parms.bad_seg_interp = 1;% 0 to replace bad segments with NaNs and ignore in final analysis, 1 to interpolate
parms.rej_bad_comp = 2;% 1 to remove all bad components, 2 to keep eye-movement components in

% dRSA parameters
parms.MNN = 0;
parms.smoothNeuralRDM = 5;% Smoothing of neural RDM in samples. Has to be odd number, because includes centre point!  
parms.smoothModelRDM = 5;% same for model RDM
parms.similarity = 0;% 0 = correlation, 1 = principal component regression
parms.nPCRcomps = 75;% in case of pcr, maximum amount of PCR components to regress out

% temporal subsampling parameters
parms.nstim = 180; % # of pseudo-stimuli to cut out of movie for each subsampling iteration 
parms.stimlen = 10;% length of pseudo-stimuli in seconds
parms.iterations = 3; %00;
parms.minISI = 1;% minimum inter-stimulus-interval in seconds; 0 means pseudo-stimuli can touch each other
parms.maxlatency = 5;% max latency to test with dRSA latency plots in sec

% cluster or locally
%parms.cluster = 1; %!!CHANGED BC NO CLUSTER NEEDED 
isub = 4; irep = 1; imod = 1; iroi = 1; ires = 24; igaze = 0;
parfor iroi = 1:6
    Project1917_dRSA(preproc_dir, models_dir, results_dir,  parms,isub,irep,imod,iroi,ires, igaze)
end % parfor iroi = 1:6

function Project1917_dRSA(preproc_dir, models_dir, results_dir,  parms,isub,irep,imod,iroi,ires, igaze)
if ires == 12 && imod == 1% 12Hz models don't exist for pixelwise
    return
end

% if isfield(parms,'cluster')
%     % set paths
%     rootdir = '//mnt/storage/tier2/morwur/Projects/INGMAR/Project1917';
%     % start up Fieldtrip
%     addpath('//mnt/storage/tier2/morwur/Projects/INGMAR/toolboxes/fieldtrip-20231220');
%     warning('off')% we don't see warnings on the cluster anyway and might take extra time to print
% else
%     % set paths
%     rootdir = '\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\Project1917';
%     % start up Fieldtrip
%     addpath('\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\toolboxes\fieldtrip-20231220')
% end

% start up Fieldtrip
%ft_defaults

% input and output folders
indirMEG = sprintf('%s%ssub-%03d%spreprocessing', preproc_dir,filesep,isub,filesep);
if igaze == 1% gaze-dependent models
    indirMOD = sprintf('%s%ssub-%03d%smodels',models_dir,filesep,isub,filesep);    
else% gaze-invariant models
    % indirMOD = sprintf('%s%smodels',models_dir,filesep);
    indirMOD = models_dir;
end

if parms.similarity == 0
    simstring = 'corr';
elseif parms.similarity == 1
    simstring = ['pcr_' num2str(parms.nPCRcomps) 'comps'];
end

outdir = sprintf('%s%sdRSA%s%s%s%dHz_%dstim_%dsec_%diter_%dMNN%ssub%03d',...
    results_dir,filesep,filesep,simstring,filesep,parms.fsNew,parms.nstim,parms.stimlen,parms.iterations,parms.MNN,filesep,isub)
if ~exist(outdir,'dir')
    mkdir(outdir);
end

% fn2save = sprintf('%s%cdRSA_%s_%s_%dHz_rep%d_gazedep%d_gazerad%d',
% outdir, filesep, parms.ROInames{iroi}, parms.modelnames{imod}, ires, irep, igaze, parms.gazeradius);
%!!CHANGED not sure what it does

% load MEG and model data
% MEG data
% fn2load = sprintf('%s%ssub%03d_%s_%dHz_MNN%d_badmuscle%d_badlowfreq%d_badsegint%d_badcomp%d',...
%     indirMEG,filesep,isub,parms.ROInames{iroi},parms.fsNew,parms.MNN,parms.rej_bad_muscle,parms.rej_bad_lowfreq,parms.bad_seg_interp,parms.rej_bad_comp);
fn2load = sprintf('%s%ssub%03d_%s_%dHz.',...
     indirMEG,filesep,isub,parms.ROInames{iroi},parms.fsNew);

% load(fn2load,'data_final');
load(fn2load)
data_final = data.trial % my preproc5 didn't have data_final, so I am doing it now
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
    count = count + 1;
    if igaze == 1% gaze dependent models
        fn2load = sprintf('%s%sProject1917_%s_sub%03d_run%02d_movie%dHz_gazerad%d',indirMOD,filesep,parms.modelnames{imod},isub,irun,ires,parms.gazeradius);
    else% gaze invariant models
        fn2load = sprintf('%s%sProject1917_%s_run%02d_movie%dHz',indirMOD,filesep,parms.modelnames{imod},count,ires);      
    end
    load(fn2load,'vecrep','fsVid','tVid');

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

% There is still ~3 sec overlap remaining between first and second movie part, but exact sample number might be slightly different due to several resampling steps
% Here we check what the remaining overlap is, store this, and remove it from the start of the second movie part
% [matchCORR,matchID] = sort(corr(single(vecrep_all{1}(:,end)),single(vecrep_all{2}(:,1:4*parms.fsNew))),'descend');
% % in case of several with the same max correlation, pick the last one
% maxmatch = max(matchID(matchCORR == matchCORR(1)));
% matchCORR = matchCORR(matchID == maxmatch);
% matchID = matchID(matchID == maxmatch);
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

% in some rare samples, eye-tracker data is missing, or at least NaN, which became zeros in uint8, so change those to background color with some noise
% so a correlation never results in a nan result
% for itime = 1:size(dataMOD,2)
%     for ipix = 1:size(dataMOD,1)
%         if dataMOD(ipix,itime) == 0
%             dataMOD(ipix,itime) = 101+round(2*rand(1,1));
%         end
%     end
% end

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

% quick whether our mask is correct, and our onsetIDs don't overlap with the mask
% if ~isfield(parms,'cluster')
%     selectedIDs = zeros(parms.iterations,framenumtot);
%     for iter = 1:parms.iterations
%         for istim = 1:parms.nstim
%             selectedIDs(iter,onsetIDiter(iter,istim):onsetIDiter(iter,istim)+parms.stimlen*parms.fsNew-1) = 1;
%         end
%     end
%     figure;
%     subplot(2,1,1);
%     imagesc(repmat(mask,parms.iterations,1));
%     title('mask');
%     subplot(2,1,2);
%     imagesc(selectedIDs)
%     title('pseudo-stimuli')
%     clear selectedIDs
% end

% compute dRSA across iterations
% set some parameters
framenum = parms.stimlen*parms.fsNew;
tRange=parms.maxlatency*parms.fsNew;% in samples
latencytime = -parms.maxlatency:1/parms.fsNew:parms.maxlatency;
pseudotime = 0:1/parms.fsNew:framenum/parms.fsNew-1/parms.fsNew;

dRSAlatency = zeros(parms.iterations,length(latencytime),'single');
parfor iter = 1:parms.iterations

    clc;
    disp(['Running dRSA iteration ' num2str(iter) ' of ' num2str(parms.iterations)]);

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
        modelRDM(:,:,iframe) = 1 - corr(dataMODiter);

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

    %     % scale once across all time points to keep temporal structure intact
    %     neuralRDMvec = reshape(rescale(reshape(neuralRDMvec,size(neuralRDMvec,1)*size(neuralRDMvec,2),1),0,2),size(neuralRDMvec,1),size(neuralRDMvec,2));
    %     modelRDMvec = reshape(rescale(reshape(modelRDMvec,size(modelRDMvec,1)*size(modelRDMvec,2),1),0,2),size(modelRDMvec,1),size(modelRDMvec,2));
    %
    %     % neuralRDM is already rescaled above across the whole time range, but it still needs to be centered per individual time point for e.g., PCA
    %     neuralRDMvec = neuralRDMvec - repmat(nanmean(neuralRDMvec),size(neuralRDMvec,1),1);
    %     modelRDMvec = modelRDMvec - repmat(nanmean(modelRDMvec),size(modelRDMvec,1),1);

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
fn2save = sprintf("%s%sdRSA_%s_sub%03d_%s_rep%d_%dHz", results_dir, filesep, simstring, isub, parms.modelnames{imod}, irep, parms.fsNew)
% save dRSA results
save(fn2save,'dRSA','latencytime');
end %EOF


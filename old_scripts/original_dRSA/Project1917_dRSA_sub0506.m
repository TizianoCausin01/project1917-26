function Project1917_dRSA_sub0506(cfg,isub,imod,icon,iroi)

if isfield(cfg,'cluster')
    addpath('//mnt/storage/tier2/morwur/Projects/INGMAR/ActionPrediction/toolboxes/fieldtrip-20191113');
    rootdir = '//mnt/storage/tier2/morwur/Projects/INGMAR/Project1917';
    warning('off')% we don't see warnings on the cluster anyway and might take extra time to print
else
    addpath('\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\ActionPrediction\toolboxes\fieldtrip-20191113');
    rootdir = '\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\Project1917';
end

% start up Fieldtrip
ft_defaults

%% input and output folders
addpath(genpath(rootdir));
indirMEG = fullfile(rootdir,'data','preprocessing',['sub' num2str(isub)]);
indirMOD = fullfile(rootdir,'data','models');

if cfg.similarity == 0
    simstring = 'corr';
elseif cfg.similarity == 1
    simstring = ['pcr_' num2str(cfg.nPCRcomps) 'comps'];
end

outdir = sprintf('%s%sresults%sdRSA%s%s_%dHz_%dstim_%dsec_%diterations_%dMNN_strictICA',rootdir,filesep,filesep,filesep,simstring,cfg.fsNew,cfg.nstim,cfg.stimlen,cfg.iterations,cfg.MNN);
if ~exist(outdir,'dir')
    mkdir(outdir);
end

fn2save = sprintf('%s%cSUB%02d_%s_%s_%s', outdir, filesep, isub, cfg.ROInames{iroi}, cfg.modelnames{imod}, cfg.conditions{icon});

%% load MEG and model data
% MEG data
if icon == 3
    fn2load = sprintf('%s%spreproc-data-final-MNN%d-%s-%dhz-sub%d_allrunscombined',indirMEG,filesep,cfg.MNN,cfg.ROInames{iroi},cfg.fsNew,isub);
    load(fn2load,'data_final');
else
    fn2load = sprintf('%s%spreproc-data-final-MNN%d-%s-%dhz-sub%d_allruns',indirMEG,filesep,cfg.MNN,cfg.ROInames{iroi},cfg.fsNew,isub);
    load(fn2load,'data_final');
    data_final = data_final{icon};
end

% model data
fn2load = sprintf('%s%sProject1917_modelvecrep_%s_part1_sdsf40',indirMOD,filesep,cfg.modelnames{imod});
load(fn2load,'vecrep');

% if after averaging over runs a sensor still shows nans (or zeros after rescaling to uint16), remove it from both movie parts
badchan = all(isnan(data_final),2) | all(data_final == 0,2);

% cut the MEG data at the end, i.e., we
% stored up until 4 sec (inter-trigger-interval) after the last trigger,
% but the movie stopped a bit earlier
dataMEG = data_final(:,1:size(vecrep,2));
dataMEG(badchan,:) = [];
clear data_final

% now concatenate the models
dataMOD = vecrep;
clear vecrep

%% create indices for random subsampling of cfg.nstim pseudo-stimuli over cfg.iterations iterations
% make sure rand numbers are different for each subject, each ROI, and each time this script is ran at a different time and date
rng(isub*10^7+iroi*10^6+second(datetime('now'))*10^4);
framenumtot = size(dataMEG,2);

% create mask with samples to ignore for random subsampling of pseudo-stimuli
% use middle of 2 movie parts and any NaNs in the data
mask = false(1,framenumtot);

% find any NaNs in MEG data and count backwards to ignore all onset indices
% that would cause a pseudo-trial to overlap with the NaNs
badsegs = find(dataMEG(1,:)==0);
mask(badsegs-cfg.stimlen*cfg.fsNew+2:badsegs) = true;

% check if requested pseudo-trials fit in data length and otherwise give error
if cfg.nstim*cfg.stimlen*cfg.fsNew + (cfg.nstim-1)*cfg.minISI*cfg.fsNew > framenumtot
    error(['The requested combination of ' num2str(cfg.nstim) ' pseudo-trials of ' num2str(cfg.stimlen) ' sec and a minimum ISI of ' num2str(cfg.minISI) ' sec does not fit in the data'])
end

% cumulative sum of pseudo-stimulus length
stimlencumsum = cumsum(cfg.stimlen*cfg.fsNew*ones(cfg.nstim-1,1));

% cumulative sum of minimum inter-stimulus-interval
minISIcumsum = cumsum(cfg.minISI*cfg.fsNew*ones(cfg.nstim-1,1));

% total length in frames of inter-stimulus-interval to divide in (nstim-1) parts
ISItotal = framenumtot - cfg.stimlen*cfg.nstim*cfg.fsNew - minISIcumsum(end);

% initialize onsetIDs = start times [iterations X stimuli];
onsetIDiter = zeros(cfg.iterations,cfg.nstim);

% loop over iterations to create onsetIDs
for iter = 1:cfg.iterations

    constraint = 1;% create new onsetIDs for this iteration until non overlap anywhere with the mask
    while constraint

        onsetID = rand(cfg.nstim+1,1);% create vector of onsetIDs
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
if ~isfield(cfg,'cluster')
    selectedIDs = zeros(cfg.iterations,framenumtot);
    for iter = 1:cfg.iterations
        for istim = 1:cfg.nstim
            selectedIDs(iter,onsetIDiter(iter,istim):onsetIDiter(iter,istim)+cfg.stimlen*cfg.fsNew-1) = 1;
        end
    end
    figure;
    subplot(2,1,1);
    imagesc(repmat(mask,cfg.iterations,1));
    title('mask');
    subplot(2,1,2);
    imagesc(selectedIDs)
    title('pseudo-stimuli')
end

%% compute dRSA across iterations
% set some parameters
framenum = cfg.stimlen*cfg.fsNew;
tRange=cfg.maxlatency*cfg.fsNew;% in samples
latencytime = -cfg.maxlatency:1/cfg.fsNew:cfg.maxlatency;
pseudotime = 0:1/cfg.fsNew:framenum/cfg.fsNew-1/cfg.fsNew;

dRSAlatency = zeros(cfg.iterations,length(latencytime),'single');
for iter = 1:cfg.iterations

    clc;
    disp(['Running dRSA iteration ' num2str(iter) ' of ' num2str(cfg.iterations)]);

    % loop over frames of pseudo-stimuli and compute RDM at each frame
    neuralRDM = zeros(cfg.nstim,cfg.nstim,framenum,'single');
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
    neuralRDMvec = zeros((cfg.nstim*cfg.nstim-cfg.nstim)/2,framenum,'single');
    modelRDMvec = neuralRDMvec;
    for iframe = 1:framenum
        neuralRDMvec(:,iframe) = squareform(tril(squeeze(neuralRDM(:,:,iframe)),-1));
        modelRDMvec(:,iframe) = squareform(tril(squeeze(modelRDM(:,:,iframe)),-1));
    end

    % temporally smooth RDMs
    if cfg.smoothNeuralRDM
        neuralRDMvec = ft_preproc_smooth(neuralRDMvec,cfg.smoothNeuralRDM);
    end
    if cfg.smoothModelRDM
        modelRDMvec = ft_preproc_smooth(modelRDMvec,cfg.smoothModelRDM);
    end

%     % scale once across all time points to keep temporal structure intact
%     neuralRDMvec = reshape(rescale(reshape(neuralRDMvec,size(neuralRDMvec,1)*size(neuralRDMvec,2),1),0,2),size(neuralRDMvec,1),size(neuralRDMvec,2));
%     modelRDMvec = reshape(rescale(reshape(modelRDMvec,size(modelRDMvec,1)*size(modelRDMvec,2),1),0,2),size(modelRDMvec,1),size(modelRDMvec,2));
% 
%     % neuralRDM is already rescaled above across the whole time range, but it still needs to be centered per individual time point for e.g., PCA
%     neuralRDMvec = neuralRDMvec - repmat(nanmean(neuralRDMvec),size(neuralRDMvec,1),1);
%     modelRDMvec = modelRDMvec - repmat(nanmean(modelRDMvec),size(modelRDMvec,1),1);
        
    % run dRSA
    if cfg.similarity == 0% simple correlation
        
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

% save dRSA results
save(fn2save,'dRSA','latencytime');
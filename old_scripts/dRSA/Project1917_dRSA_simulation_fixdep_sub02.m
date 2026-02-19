function Project1917_dRSA_simulation_fixdep_sub02(cfg,isim,imod,isub,irep)

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
indirMOD = fullfile(rootdir,'data',['SUB0' num2str(isub)],'fixdep_models');

if cfg.similarity == 0
    simstring = 'corr';
elseif cfg.similarity == 1
    simstring = ['pcr_' num2str(cfg.nPCRcomps) 'comps'];
end

outdir = sprintf('%s%sresults%sdRSA%ssim_%s_%dHz_%dstim_%dsec_%diterations_fixdep',rootdir,filesep,filesep,filesep,simstring,cfg.fsNew,cfg.nstim,cfg.stimlen,cfg.iterationsSIM);
if ~exist(outdir,'dir')
    mkdir(outdir);
end

fn2save = sprintf('%s%cSIM_%s_TEST_%s_SUB%02d_rep%d', outdir, filesep, cfg.modelnames{isim}, cfg.modelnames{imod}, isub, irep);

%% load model data
% select current repetition
runIDs = 2*irep-1:2*irep;

% for simulation
fn2load = sprintf('%s%sProject1917_modelvecrep_%s_fixdep_rad200_run%d',indirMOD,filesep,cfg.modelnames{isim},runIDs(1));
vecrep1 = load(fn2load,'vecrep');

fn2load = sprintf('%s%sProject1917_modelvecrep_%s_fixdep_rad200_run%d',indirMOD,filesep,cfg.modelnames{isim},runIDs(2));
vecrep2 = load(fn2load,'vecrep');

% remember last sample of first movie part for later mask
part1endID = size(vecrep1.vecrep,2);

% now concatenate the models
dataSIM = cat(2,vecrep1.vecrep,vecrep2.vecrep);
clear vecrep1 vecrep2

% model data, only load if different from simulated data
if imod ~= isim
    % for simulation
    fn2load = sprintf('%s%sProject1917_modelvecrep_%s_fixdep_rad200_run%d',indirMOD,filesep,cfg.modelnames{imod},runIDs(1));
    vecrep1 = load(fn2load,'vecrep');
    
    fn2load = sprintf('%s%sProject1917_modelvecrep_%s_fixdep_rad200_run%d',indirMOD,filesep,cfg.modelnames{imod},runIDs(2));
    vecrep2 = load(fn2load,'vecrep');

    % now concatenate the models
    dataMOD = cat(2,vecrep1.vecrep,vecrep2.vecrep);
    clear vecrep1 vecrep2
else
    dataMOD = dataSIM;
end

% in some rare samples, eye-tracker data is missing, or at least NaN, which became zeros in uint8, so change those to background color with some noise
% so a correlation never results in a nan result
for itime = 1:size(dataMOD,2)
    for ipix = 1:size(dataMOD,1)
        if dataMOD(ipix,itime) == 0
           dataMOD(ipix,itime) = 101+round(2*rand(1,1));
        end
        if dataSIM(ipix,itime) == 0
           dataSIM(ipix,itime) = 101+round(2*rand(1,1));
        end
    end
end

%% create indices for random subsampling of cfg.nstim pseudo-stimuli over cfg.iterationsSIM iterations
% make sure rand numbers are different for each subject, each ROI, and each time this script is ran at a different time and date
rng(isim*10^7+imod*10^6+second(datetime('now'))*10^4);
framenumtot = size(dataSIM,2);

% create mask with samples to ignore for random subsampling of pseudo-stimuli
% use middle of 2 movie parts and any NaNs in the data
mask = false(1,framenumtot);
mask(part1endID-cfg.stimlen*cfg.fsNew+2:part1endID) = true;

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
onsetIDiter = zeros(cfg.iterationsSIM,cfg.nstim);

% loop over iterations to create onsetIDs
for iter = 1:cfg.iterationsSIM

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
selectedIDs = zeros(cfg.iterationsSIM,framenumtot);
for iter = 1:cfg.iterationsSIM
    for istim = 1:cfg.nstim
        selectedIDs(iter,onsetIDiter(iter,istim):onsetIDiter(iter,istim)+cfg.stimlen*cfg.fsNew-1) = 1;
    end
end
figure;
subplot(2,1,1);
imagesc(repmat(mask,cfg.iterationsSIM,1));
title('mask');
subplot(2,1,2);
imagesc(selectedIDs)
title('pseudo-stimuli')

%% compute dRSA across iterations
% set some parameters
framenum = cfg.stimlen*cfg.fsNew;
tRange=cfg.maxlatency*cfg.fsNew;% in samples
latencytime = -cfg.maxlatency:1/cfg.fsNew:cfg.maxlatency;
pseudotime = 0:1/cfg.fsNew:framenum/cfg.fsNew-1/cfg.fsNew;

dRSAlatency = zeros(cfg.iterationsSIM,length(latencytime),'single');
for iter = 1:cfg.iterationsSIM

    clc;
    disp(['Running dRSA iteration ' num2str(iter) ' of ' num2str(cfg.iterationsSIM)]);

    % loop over frames of pseudo-stimuli and compute RDM at each frame
    neuralRDM = zeros(cfg.nstim,cfg.nstim,framenum,'single');
    modelRDM = neuralRDM;
    for iframe = 1:framenum
        
        % select current frame for all pseudo-stimuli
        frameIDs = onsetIDiter(iter,:)+iframe-1;

        dataMEGiter = single(dataSIM(:,frameIDs));
        dataMODiter = single(dataMOD(:,frameIDs));

        % center data before computing RDMs (recommended by cosmoMVPA)
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
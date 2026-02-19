function Project1917_dRSA_inclMNN(cfg,isub,imod)

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

outdir = sprintf('%s%sresults%sdRSA%s%s_%dHz_%dstim_%dsec_%diterations_3MNN',rootdir,filesep,filesep,filesep,simstring,cfg.fsNew,cfg.nstim,cfg.stimlen,cfg.iterations);
if ~exist(outdir,'dir')
    mkdir(outdir);
end

fn2save = sprintf('%s%cSUB%02d_%s_%s', outdir, filesep, isub, cfg.ROInames{cfg.ROI}, cfg.modelnames{imod});

%% load MEG and model data
% MEG data
fn2load = sprintf('%s%spreproc-data-final-MNN%d-%s-%dhz-sub%d_allruns',indirMEG,filesep,cfg.MNN,cfg.ROInames{cfg.ROI},cfg.fsNew,isub);
load(fn2load,'data_final');

% model data
fn2load = sprintf('%s%sProject1917_modelvecrep_%s_part1',indirMOD,filesep,cfg.modelnames{imod});
vecrep1 = load(fn2load,'vecrep');

fn2load = sprintf('%s%sProject1917_modelvecrep_%s_part2',indirMOD,filesep,cfg.modelnames{imod});
vecrep2 = load(fn2load,'vecrep');

% % if a sensor still shows nans (or zeros after rescaling to uint16), remove it from both movie parts
% badchan = false(2,size(data_final{1},1));
% for ipart = 1:2
%     
%     badchan(ipart,:) = all(isnan(data_final{ipart}),2) | all(data_final{ipart} == 0,2);
%     
% end
% 
% badchan = logical(sum(badchan));

% concatenate the two movie parts and cut the MEG data at the end, i.e., we
% stored up until 4 sec (inter-trigger-interval) after the last trigger,
% but the movie stopped a bit earlier
dataMEG = cat(2,data_final{1}(:,1:size(vecrep1.vecrep,2),:),data_final{2}(:,1:size(vecrep2.vecrep,2),:));
% dataMEG(badchan,:) = [];
clear data_final

% remember last sample of first movie part for later mask
part1endID = size(vecrep1.vecrep,2);

% now concatenate the models
dataMOD = cat(2,vecrep1.vecrep,vecrep2.vecrep);
clear vecrep1 vecrep2

%% create indices for random subsampling of cfg.nstim pseudo-stimuli over cfg.iterations iterations
% make sure rand numbers are different for each subject, each ROI, and each time this script is ran at a different time and date
rng(isub*10^7+cfg.ROI*10^6+second(datetime('now'))*10^4);
framenumtot = size(dataMEG,2);

% create mask with samples to ignore for random subsampling of pseudo-stimuli
% use middle of 2 movie parts and any NaNs in the data
mask = false(1,framenumtot);
mask(part1endID-cfg.stimlen*cfg.fsNew+2:part1endID) = true;

% find any NaNs in MEG data and count backwards to ignore all onset indices
% that would cause a pseudo-trial to overlap with the NaNs
badsegs = find(all(isnan(dataMEG(1,:,:)),3));
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
    clear selectedIDs
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
    
    % select pseudo-stimuli in neural data
    frameIDs = onsetIDiter(iter,:);
    
    dataMEGiter = zeros(cfg.nstim,size(dataMEG,1),framenum,3,'single');
    for istim = 1:cfg.nstim
        dataMEGiter(istim,:,:,:) = dataMEG(:,frameIDs(istim):frameIDs(istim)+framenum-1,:);
    end
    
    %% multivariate noise normalisation with shrinkage (Guggenmos et al. 2018 NeuroImage), using epoch method
    n_sensors = size(dataMEG,1);% # of sensors for this ROI
    n_stim = cfg.nstim;% # of psuedo-stimuli
    n_time = framenum;
    
    sigma_perstim = nan(n_stim, n_sensors, n_sensors);
    for istim = 1:n_stim
        % compute sigma for each time point
        sigma_pertime = nan(n_time, n_sensors, n_sensors);
        for itime = 1:n_time
            data2use = squeeze(dataMEGiter(istim,:,itime,:));
            data2use = double(data2use);
            data2use(any(isnan(data2use),2),:) = [];
            if size(data2use,2) == 1 || isempty(data2use)
                continue
            end
            sigma_pertime(itime, :, :) = cov1para(data2use');
        end
        sigma_perstim(istim, :, :) = mean(sigma_pertime, 1, 'omitnan');% average across time
    end
    sigma = squeeze(mean(sigma_perstim, 1));  % average across conditions
    sigma_inv = sigma^-0.5;
    
    % now apply to data
    for itime = 1:n_time
        for irep = 1:3
            dataMEGiter(:,:,itime,irep) = squeeze(dataMEGiter(:,:,itime,irep)) * sigma_inv;
        end
    end
    
    % average over runs
    dataMEGiter = squeeze(mean(dataMEGiter,4,'omitnan'));
    
    % center data across stimuli before computing RDMs (recommended by cosmoMVPA)
    %         dataMEGiter = dataMEGiter - mean(dataMEGiter,2);
    
    % compute RDM for current frame
    neuralRDM = zeros(cfg.nstim,cfg.nstim,framenum,'single');
    for iframe = 1:framenum
        neuralRDM(:,:,iframe) = 1 - corr(squeeze(dataMEGiter(:,:,iframe))');
    end
    
    % loop over frames of pseudo-stimuli and compute model RDM at each frame
    modelRDM = zeros(cfg.nstim,cfg.nstim,framenum,'single');
    for iframe = 1:framenum
        
        % select current frame for all pseudo-stimuli
        frameIDs = onsetIDiter(iter,:)+iframe-1;
        
        dataMODiter = single(dataMOD(:,frameIDs));
        
        % center data across stimuli before computing RDMs (recommended by cosmoMVPA)
        %         dataMODiter = dataMODiter - mean(dataMODiter,2);
        
        % compute RDM for current frame
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
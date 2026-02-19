function Project1917_dRSA_simulation(parms,isub,irep,imod,~,ires,igaze)

if ires == 12 && imod == 1% 12Hz models don't exist for pixelwise
    return
end

if isfield(parms,'cluster')
    % set paths
    rootdir = '//mnt/storage/tier2/morwur/Projects/INGMAR/Project1917';
    % start up Fieldtrip
    addpath('//mnt/storage/tier2/morwur/Projects/INGMAR/toolboxes/fieldtrip-20231220');
    warning('off')% we don't see warnings on the cluster anyway and might take extra time to print
else
    % set paths
    rootdir = '\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\Project1917';
    % start up Fieldtrip
    addpath('\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\toolboxes\fieldtrip-20231220')
end

% start up Fieldtrip
ft_defaults

%% input and output folders
if igaze == 1% gaze-dependent models
    indirMOD = sprintf('%s%sdata%ssub-%03d%smodels',rootdir,filesep,filesep,isub,filesep);    
else% gaze-invariant models
    indirMOD = sprintf('%s%sdata%smodels',rootdir,filesep,filesep);
end

if parms.similarity == 0
    simstring = 'corr';
elseif parms.similarity == 1
    simstring = ['pcr_' num2str(parms.nPCRcomps) 'comps'];
end

outdir = sprintf('%s%sresults%sdRSA%s%s%s%dHz_%dstim_%dsec_%diter_%dMNN%ssub%03d',...
    rootdir,filesep,filesep,filesep,simstring,filesep,parms.fsNew,parms.nstim,parms.stimlen,parms.iterations,parms.MNN,filesep,isub);
if ~exist(outdir,'dir')
    mkdir(outdir);
end

fn2save = sprintf('%s%cdRSA_SIM_%s_TEST_%s_%dHz_rep%d_gazedep%d_gazerad%d', outdir, filesep, parms.modelnames{imod}, parms.modelnames{imod}, ires, irep, igaze, parms.gazeradius);

%% load model data
% select current repetition
runIDs = 3*irep-2:3*irep;

% load model data for simulation
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

    vecrep_all{count} = vecrep;
    clear vecrep
end

% There is still ~3 sec overlap remaining between first and second movie part, but exact sample number might be slightly different due to several resampling steps
% Here we check what the remaining overlap is, store this, and remove it from the start of the second movie part
matchID = 152;% just set to 152 because determined with view-invariant pixelwise model, which should then hold for all

% now cut the overlap
vecrep_all{2}(:,1:matchID) = [];

% remember last sample of second movie part for later mask
part2endID = size(vecrep_all{1},2)+size(vecrep_all{2},2);

% now concatenate the models
dataMOD = cat(2,vecrep_all{1},vecrep_all{2},vecrep_all{3});
clear vecrep_all

% model data, only load if different from simulated data
% if imod ~= isim
%     % for simulation
%     fn2load = sprintf('%s%sProject1917_modelvecrep_%s_fixdep_rad200_run%d',indirMOD,filesep,parms.modelnames{imod},runIDs(1));
%     vecrep1 = load(fn2load,'vecrep');
% 
%     fn2load = sprintf('%s%sProject1917_modelvecrep_%s_fixdep_rad200_run%d',indirMOD,filesep,parms.modelnames{imod},runIDs(2));
%     vecrep2 = load(fn2load,'vecrep');
% 
%     % now concatenate the models
%     dataMOD = cat(2,vecrep1.vecrep,vecrep2.vecrep);
%     clear vecrep1 vecrep2
% else
dataSIM = dataMOD;
% end

% in some rare samples, eye-tracker data is missing, or at least NaN, which became zeros in uint8, so change those to background color with some noise
% so a correlation never results in a nan result
% for itime = 1:size(dataMOD,2)
%     for ipix = 1:size(dataMOD,1)
%         if dataMOD(ipix,itime) == 0
%            dataMOD(ipix,itime) = 101+round(2*rand(1,1));
%         end
%         if dataSIM(ipix,itime) == 0
%            dataSIM(ipix,itime) = 101+round(2*rand(1,1));
%         end
%     end
% end

%% create indices for random subsampling of cfg.nstim pseudo-stimuli over cfg.iterationsSIM iterations
% make sure rand numbers are different for each subject, each ROI, and each time this script is ran at a different time and date
rng(imod*10^6+second(datetime('now'))*10^4);
framenumtot = size(dataSIM,2);

% create mask with samples to ignore for random subsampling of pseudo-stimuli
% use middle of 2 movie parts and any NaNs in the data
mask = false(1,framenumtot);
mask(part2endID-parms.stimlen*parms.fsNew+2:part2endID) = true;

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
onsetIDiter = zeros(parms.iterationsSIM,parms.nstim);

% loop over iterations to create onsetIDs
for iter = 1:parms.iterationsSIM

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
if ~isfield(parms,'cluster')
    selectedIDs = zeros(parms.iterationsSIM,framenumtot);
    for iter = 1:parms.iterationsSIM
        for istim = 1:parms.nstim
            selectedIDs(iter,onsetIDiter(iter,istim):onsetIDiter(iter,istim)+parms.stimlen*parms.fsNew-1) = 1;
        end
    end
    figure;
    subplot(2,1,1);
    imagesc(repmat(mask,parms.iterationsSIM,1));
    title('mask');
    subplot(2,1,2);
    imagesc(selectedIDs)
    title('pseudo-stimuli')
    clear selectedIDs
end

%% compute dRSA across iterations
% set some parameters
framenum = parms.stimlen*parms.fsNew;
tRange=parms.maxlatency*parms.fsNew;% in samples
latencytime = -parms.maxlatency:1/parms.fsNew:parms.maxlatency;
pseudotime = 0:1/parms.fsNew:framenum/parms.fsNew-1/parms.fsNew;

dRSAlatency = zeros(parms.iterationsSIM,length(latencytime),'single');
for iter = 1:parms.iterationsSIM

    clc;
    disp(['Running dRSA iteration ' num2str(iter) ' of ' num2str(parms.iterationsSIM)]);

    % loop over frames of pseudo-stimuli and compute RDM at each frame
    neuralRDM = zeros(parms.nstim,parms.nstim,framenum,'single');
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
    neuralRDMvec = zeros((parms.nstim*parms.nstim-parms.nstim)/2,framenum,'single');
    modelRDMvec = neuralRDMvec;
    for iframe = 1:framenum
        neuralRDMvec(:,iframe) = squareform(tril(squeeze(neuralRDM(:,:,iframe)),-1));
        modelRDMvec(:,iframe) = squareform(tril(squeeze(modelRDM(:,:,iframe)),-1));
    end

    % temporally smooth RDMs
    if parms.smoothModelRDM
        neuralRDMvec = ft_preproc_smooth(neuralRDMvec,parms.smoothModelRDM);
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

% save dRSA results
save(fn2save,'dRSA','latencytime');
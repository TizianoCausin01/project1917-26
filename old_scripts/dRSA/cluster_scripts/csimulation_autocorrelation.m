close all; clearvars; clc

addpath("/home/tiziano.causin/adds_on/fieldtrip-20250114")
ft_defaults
% parms.modelnames = {"alexnet_conv_layer1", "alexnet_conv_layer4", "alexnet_conv_layer7", "alexnet_conv_layer9", "alexnet_conv_layer11", "alexnet_fc_layer2", "alexnet_fc_layer5", "resnet18_layer1", "resnet18_layer2", "resnet18_layer3", "resnet18_layer4", "resnet18_fc"}
parms.modelnames = {"real_alexnet_real_conv_layer1", "real_alexnet_real_conv_layer4", "real_alexnet_real_conv_layer7", "real_alexnet_real_conv_layer9", "real_alexnet_real_conv_layer11", "real_alexnet_real_fc_layer2", "real_alexnet_real_fc_layer5"};
parms.metric = 'corr';
models_directory = "/mnt/storage/tier2/ingdev/projects/TIZIANO/models/";
%models_directory = "/Volumes/TIZIANO/models/";
simulation_output_directory = "/mnt/storage/tier2/ingdev/projects/TIZIANO/results/simulations/";
%simulation_output_directory = "/Volumes/TIZIANO/results/simulations/";
parms.nstim = 180; % # of pseudo-stimuli to cut out of movie for each subsampling iteration. It will create an nxn RDM
parms.stimlen = 10;% length of pseudo-stimuli in seconds
parms.iterations = 100; %number of times the process will be repeated
parms.fsNew = 50;
parms.minISI = 1;% minimum inter-stimulus-interval in seconds; 0 means pseudo-stimuli can touch each other
parms.maxlatency = 5;% max latency to test with dRSA latency plots in sec


for imod_num = 1 : size(parms.modelnames,2)
    imod = parms.modelnames{imod_num};
    for irun = 1 : 3
        fn2load = sprintf('%sProject1917_%s_run%02d_movie24Hz.mat' , models_directory,imod,irun);
        load(fn2load);
        tNew = 0:1/parms.fsNew:size(vecrep,2)/fsVid-1/parms.fsNew;
        old2newID=dsearchn(tVid',tNew');
        vecrep=vecrep(:,old2newID,:);
        vecrep_total{irun}=vecrep;
        clear vecrep
    end
    vecrep_total{2}(:,1:3*parms.fsNew)=[];
    simulation_autocorrelation(vecrep_total,simulation_output_directory,imod, parms)
end


% it should work for all the models
% gets the models name and the parms and the dataDir and returns a
% results.mat for each subject in the form of dRSA_sub{3dnumber}_{model_name}_rep{irep}_{freq_of_analysis}Hz.mat in the results folder
function simulation_autocorrelation(dataMOD, output_directory, imod, parms)
%this function runs the dRSA for one subject
% takes as inputs:
% - dataMOD -> the model already at the correct frequency of analysis and
%   with the overlap between run 1 and 2 already cut
% - output_directory -> the directory in which you want your output stored
% - imod -> a char vector with the corresponding name of the model to test
% - isub -> the subject currently being analysed
% - irep -> the number of the ongoing repetition
% - parms -> a struct comprehensive of :
%   .metric
%   .stimlen, .fsNew, .iterations, .nstim, .minISI, .maxlatency,


% input and output folders

if parms.fsNew == 50
    freq = 50;
else
    freq = 24;
end
fn2save = sprintf('%s%csimulation_%s_%dHz.mat', output_directory, filesep,imod,freq);

% load eye-tracker and model data
% eye-tracker data data
%stores the size of the runs of the model to chop out supplementary data
for i=1:size(dataMOD,2)
    size_runs(i)=size(dataMOD{i},2);
end

data_final = dataMOD; % key step to have the model as data

% select current repetition

matchID=3*parms.fsNew;
for i=1:size(data_final,2)
    data_final{i}(:,size_runs(i)+1:end)=[]; %chops out supplementary data
end
dataEYE = cat(2,data_final{1},data_final{2},data_final{3});
dataMOD = cat(2,dataMOD{1:3});
part2endID = size(data_final{1},2)+size(data_final{2},2);
clear data_final
% remember last sample of second movie part for later mask

% create indices for random subsampling of cfg.nstim pseudo-stimuli over cfg.iterations iterations
% make sure rand numbers are different for each subject, and each time this script is ran at a different time and date
rng(second(datetime('now'))*10^4);
framenumtot = size(dataEYE,2);

% create mask with samples to ignore for random subsampling of pseudo-stimuli, use middle of movie parts 2 and 3 and any NaNs in the data
mask = false(1,framenumtot);
mask(part2endID-parms.stimlen*parms.fsNew+2:part2endID) = true;

% find any NaNs in MEG data and count backwards to ignore all onset indices
% that would cause a pseudo-trial to overlap with the NaNs, uint16 operation changes NaNs to zeros
badsegs = find(isnan(dataEYE(1,:)));
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
for iter = 1:parms.iterations %creates a series of n random iterations to compute the dRSA n times with n different 180 segments (n=parms.iteration)
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
    selectedIDs = zeros(parms.iterations,framenumtot);
    for iter = 1:parms.iterations
        for istim = 1:parms.nstim
            selectedIDs(iter,onsetIDiter(iter,istim):onsetIDiter(iter,istim)+parms.stimlen*parms.fsNew-1) = 1;
        end
    end
    clear selectedIDs
end

% compute dRSA across iterations
% set some parameters
framenum = parms.stimlen*parms.fsNew; %num of frames in each segment, corresponds to 10 sec
tRange=parms.maxlatency*parms.fsNew;% in samples
latencytime = -parms.maxlatency:1/parms.fsNew:parms.maxlatency;
pseudotime = 0:1/parms.fsNew:framenum/parms.fsNew-1/parms.fsNew;

dRSAlatency = zeros(parms.iterations,length(latencytime),'single');
for iter = 1:parms.iterations

    disp(['Running dRSA iteration ' num2str(iter) ' of ' num2str(parms.iterations) imod]);
    drawnow

    % loop over frames of pseudo-stimuli and compute RDM at each frame
    gazeRDM = zeros(parms.nstim,parms.nstim,round(framenum),'single'); %changed ingmar
    modelRDM = gazeRDM;
    for iframe = 1:framenum

        % select current frame for all pseudo-stimuli
        frameIDs = onsetIDiter(iter,:)+iframe-1;

        dataGAZEiter = single(dataEYE(:,frameIDs));
        dataMODiter = single(dataMOD(:,frameIDs));

        % compute RDM for current frame
        gazeRDM(:,:,iframe) = 1 - corr(dataGAZEiter); 
        % depending on whether I'm using a saliency map or a gaze simulation
        if parms.metric == 'corr'
            modelRDM(:,:,iframe) = 1 - corr(dataMODiter);
        elseif parms.metric == 'dist'
            modelRDM(:,:,iframe) = dist(dataMODiter);
        end
    end % frame loop

    % take lower triangle from square RDM and vectorize
    eyeRDMvec = zeros((parms.nstim*parms.nstim-parms.nstim)/2,round(framenum),'single');
    modelRDMvec = eyeRDMvec;
    for iframe = 1:framenum
        eyeRDMvec(:,iframe) = squareform(tril(squeeze(gazeRDM(:,:,iframe)),-1));
        modelRDMvec(:,iframe) = squareform(tril(squeeze(modelRDM(:,:,iframe)),-1));
    end

    % run dRSA
    %  gaze - model correlation
    dRSAiter = corr(modelRDMvec,eyeRDMvec,'Type','Spearman'); %!! changed bc I'm using euclidean distance for eyetracking

    % slice gaze vectors per model time point, and then stack, to create gaze - model latency
    dRSAstack = zeros(length(pseudotime),length(latencytime));
    for iModelTime = 1:length(pseudotime)

        timeidx = iModelTime - tRange:iModelTime + tRange;
        NotInVid = logical((timeidx < 1)+(timeidx > length(pseudotime)));
        timeidx(NotInVid) = 1;%remove indices that fall before or after video

        slice = squeeze(dRSAiter(iModelTime,round(timeidx)));
        slice(NotInVid) = NaN;%remove indices that fall before or after video
        dRSAstack(iModelTime,:) = slice;
    end
    lagplot(:,:,iter)=dRSAstack; %!! to create lagplot
    %Average over video time
    dRSAlatency(iter,:) = squeeze(mean(dRSAstack,1,'omitnan'));

end
%figure; imagesc(mean(lagplot,3, 'omitnan')) %!! to visualize the lagplot
% average over iterations
dRSA = squeeze(mean(dRSAlatency));
standard_err_simulation = squeeze(std(dRSAlatency)/parms.iterations);
%figure;
%plot(latencytime,dRSA)
% save dRSA results
save(fn2save,'dRSA','latencytime','standard_err_simulation');
end
